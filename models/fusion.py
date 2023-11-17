import torch
from torch import nn
from einops import rearrange, repeat
from models.vit import FeedForward


class MaskAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., seqLs= []):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.seqLs = seqLs

    def get_mask(self, modality_labels):
        batch_size, num_modalities = modality_labels.shape
        device = modality_labels.device
        total_length = sum(self.seqLs)
    
        # Initialize the mask with zeros (will be filled with -inf for missing modalities)
        mask = torch.zeros((batch_size, total_length, total_length), device=device)
        
        # Calculate the start and end indices for each modality
        end_indices = torch.cumsum(torch.tensor(self.seqLs, device=device), dim=0)
        start_indices = torch.cat((torch.tensor([0], device=device), end_indices[:-1]))
        
        # Iterate over each modality and update the mask
        for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
            # Check where the modality is missing (0 in modality_labels)
            modality_missing = modality_labels[:, i].unsqueeze(1).unsqueeze(2) == 0
            
            # Update the mask for missing modalities
            # Expand the mask to cover the entire rows and columns for the missing modality
            mask[:, start_idx:end_idx, :] = mask[:, start_idx:end_idx, :].masked_fill(modality_missing, float('-inf'))
            mask[:, :, start_idx:end_idx] = mask[:, :, start_idx:end_idx].masked_fill(modality_missing, float('-inf'))
        
        return mask

    def forward(self, x, m_labels):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        mask = self.get_mask(m_labels).unsqueeze(1) # [B, 1, N1, N2]
        mask = repeat(mask, 'b 1 n1 n2 -> b h n1 n2', h = dots.shape[1]) # broadcast to h heads
        dots += mask

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class FusionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., seqLs=[]):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MaskAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, seqLs=seqLs),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, m_labels):
        for attn, ff in self.layers:
            x = attn(x, m_labels) + x
            x = ff(x) + x

        return self.norm(x)

 
    
class FusionNetwork(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., seqLs=[]):
        super().__init__()
        self.modality_emb = nn.ParameterList([nn.Parameter(torch.randn(1, 1, dim)) for i in range(4)])
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = FusionTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, seqLs)
        total_seqL = sum(seqLs)
        self.pos_embedding = nn.Parameter(torch.randn(1, total_seqL, dim))

    def forward(self, feature_list, m_labels):
        # add modality embeddings
        for i in range(4):
            b, n, _ = feature_list[i].shape # [B, N, C]
            modality_emb = repeat(self.modality_emb[i], '1 1 d -> b n d', b = b, n = n)
            feature_list[i] += modality_emb
        features = torch.cat(feature_list, dim=1) # [B, total_seqL, C]
        # add positional embeddings, optional
        features += self.pos_embedding
        features = self.dropout(features)
        features = self.transformer(features, m_labels) # [B, total_seqL, C]
 
        return features
