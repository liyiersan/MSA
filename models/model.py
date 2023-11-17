from torch import nn
from models.vit import ViT
from models.fusion import FusionNetwork

class MLP(nn.Module):
    def __init__(self, dim_in, dim_mlp, dim_out, layer_num):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(layer_num-1):
            in_ch = dim_in if i == 0 else dim_mlp
            self.layers.append(nn.Sequential([
                nn.Linear(in_ch, dim_mlp),
                nn.LeakyReLU()
            ]))
        in_ch = dim_mlp if layer_num > 1 else dim_in
        self.layers.append(nn.Linear(in_ch, dim_out))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    


class PredictiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # image encoder
        self.oct_vit = ViT(**config['model_cfg']['image'])
        self.ultrasound_vit = ViT(**config['model_cfg']['image'])
        self.slo_vit = ViT(**config['model_cfg']['image'])

        # classification head
        self.oct_cls_head = MLP(**config['model_cfg']['cls'])
        self.ultrasound_cls_head = MLP(**config['model_cfg']['cls'])
        self.slo_cls_head = MLP(**config['model_cfg']['cls'])

        self.text_projector = MLP(**config['model_cfg']['text'])

        # to save resource, a shared img_projector or cls_head MLP is also acceptable
        self.oct_projector = MLP(**config['model_cfg']['projector'])
        self.ultrasound_projector = MLP(**config['model_cfg']['projector'])
        self.slo_projector = MLP(**config['model_cfg']['projector'])

        # fusion network
        self.fusion = FusionNetwork(**config['model_cfg']['fusion'])

        # prediction head
        self.predict_head = MLP(**config['model_cfg']['predict'])
        
    def forward(self, inputs):
        text_features, oct, ultrasound, slo, modality_labels = inputs

        oct_features = self.oct_vit(oct)
        ultrasound_features = self.ultrasound_vit(ultrasound)
        slo_features = self.slo_vit(slo)

        oct_cls = self.oct_cls_head(oct_features.mean(dim=1)) # [B, N_class]
        ultrasound_cls = self.ultrasound_cls_head(ultrasound_features.mean(dim=1)) # [B, N_class]
        slo_cls = self.slo_cls_head(slo_features.mean(dim=1)) # [B, N_class]

        # poject to same dimension
        text_features = self.text_projector(text_features).unsqueeze(1) # [B, 1, C]
        oct_features = self.oct_projector(oct_features)
        ultrasound_features = self.ultrasound_projector(ultrasound_features)
        slo_features = self.slo_projector(slo_features)

        fused_features = self.fusion([text_features, oct_features, ultrasound_features, slo_features], modality_labels)
        predicts = self.predict_head(fused_features.mean(dim=1)) # [B, 1]

        return oct_cls, ultrasound_cls, slo_cls, predicts
    

    
