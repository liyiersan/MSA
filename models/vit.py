import torch
from torch import nn
'''
modified from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
'''
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qkv_bias=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., qkv_bias = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, qkv_bias=qkv_bias),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    """
        Vision Transformer (ViT) for image feature extraction, without avg pooling for features
        Args:
        image_size: Size of the input images.
        patch_size: Size of the patches to be extracted from the images.
        dim: Dimensionality of the transformer.
        depth: Number of transformer layers.
        heads: Number of attention heads.
        mlp_dim: Dimensionality of the hidden layer in feedforward network.
        channels: Number of input image channels.
        dim_head: Dimensionality of each attention head.
        dropout: Dropout rate in transformer.
        emb_dropout: Dropout rate for patch embeddings.
        qkv_bias: Use bias in qkv projection.
        pretrain_path: Path of pretrained vit.
    """
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., qkv_bias=False, pretrain_path=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # nn.LayerNorm(patch_dim),
            # nn.Linear(patch_dim, dim),
            # nn.LayerNorm(dim),

            # use conv2d to fit weight file
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qkv_bias)

        if pretrain_path is not None:
            self.load_pretrain(pretrain_path)


    def load_pretrain(self, pretain_path):
        """
        load the jax pretrained weights from timm, note that we remove many unnecessary components (e.g., mlp_head) 
        
        weights can be downloaded from here: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-vitjx
        you can download various pretriained weights and adjust your codes to fit them

        ideas from https://github.com/Sebastian-X/vit-pytorch-with-pretrained-weights/blob/master/tools/trans_weight.py

        weights mapping as follows:
        
        timm_jax_vit_base                           self

        pos_embed                                   pos_embedding
        patch_embed.proj.weight                     to_patch_embedding.0.weights
        patch_embed.proj.bias                       to_patch_embedding.0.bias
        cls_token                                   cls_token
        norm.weight                                 transformer.norm.weight
        norm.bias                                   transformer.norm.bias

                            -----------Attention Layer-------------
        blocks.0.norm1.weight                       transformer.layers.0.0.norm.weight
        blocks.0.norm1.bias                         transformer.layers.0.0.norm.bias
        blocks.0.attn.qkv.weight                    transformer.layers.0.0.to_qkv.weight
        blocks.0.attn.qkv.bias                      transformer.layers.0.0.to_qkv.bias
        blocks.0.attn.proj.weight                   transformer.layers.0.0.to_out.0.weight
        blocks.0.attn.proj.bias                     transformer.layers.0.0.to_out.0.bias
                            -----------MLP Layer-------------
        blocks.0.norm2.weight                       transformer.layers.0.1.net.0.weight
        blocks.0.norm2.bias                         transformer.layers.0.1.net.0.bias
        blocks.0.mlp.fc1.weight                     transformer.layers.0.1.net.1.weight
        blocks.0.mlp.fc1.bias                       transformer.layers.0.1.net.1.bias
        blocks.0.mlp.fc2.weight                     transformer.layers.0.1.net.4.weight
        blocks.0.mlp.fc2.bias                       transformer.layers.0.1.net.4.bias
                .                                                      .
                .                                                      .
                .                                                      .
        """
        jax_dict = torch.load(pretain_path, map_location='cpu')
        new_dict = {}

        def add_item(key, value):
            key = key.replace('blocks', 'transformer.layers')
            new_dict[key] = value
            
        for key, value in jax_dict.items():
            if key == 'cls_token':
                new_dict[key] = value
            
            elif 'norm1' in key:
                new_key = key.replace('norm1', '0.norm')
                add_item(new_key, value)
            elif 'attn.qkv' in key:
                new_key = key.replace('attn.qkv', '0.to_qkv')
                add_item(new_key, value)
            elif 'attn.proj' in key:
                new_key = key.replace('attn.proj', '0.to_out.0')
                add_item(new_key, value)
            elif 'norm2' in key:
                new_key = key.replace('norm2', '1.net.0')
                add_item(new_key, value)
            elif 'mlp.fc1' in key:
                new_key = key.replace('mlp.fc1', '1.net.1')
                add_item(new_key, value)
            elif 'mlp.fc2' in key:
                new_key = key.replace('mlp.fc2', '1.net.4')
                add_item(new_key, value)
            elif 'patch_embed.proj' in key:
                new_key = key.replace('patch_embed.proj', 'to_patch_embedding.0')
                add_item(new_key, value)
            
            elif key == 'pos_embed':
                add_item('pos_embedding', value)
            elif key == 'norm.weight':
                add_item('transformer.norm.weight', value)
            elif key == 'norm.bias':
                add_item('transformer.norm.bias', value)
            
        self.load_state_dict(new_dict, strict=True)

    def forward(self, img):
        x = self.to_patch_embedding(img) # [B, C, H, W]
        x = x.flatten(2).transpose(1,2) # [B, N, C]
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x) # [B, N+1, C]

        return x[:,1:,:]
    
if __name__ == '__main__':
    load = True
    vit = ViT(image_size=224, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072, dim_head=64, qkv_bias=True, pretrain_path='vit_base_patch16_224.pth' if load else None)
    for k,v in vit.state_dict().items():
        print(k)
