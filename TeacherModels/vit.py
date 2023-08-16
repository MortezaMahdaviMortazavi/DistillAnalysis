import torchvision.models as models
import torch.nn as nn
import torch

class VisionTransformer(nn.Module):
    def __init__(self,transformer_name,num_classes):
        super().__init__()
        assert transformer_name in ['swin','vit']
        if transformer_name == 'swin':
            modules = list(models.swin_b(pretrained=True).children())[:-2]
            in_features = 1024
        elif transformer_name == 'vit':
            modules = list(models.vit_b_16(pretrained=True).children())[:-2]
            in_features = 768

        self.transformer = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=in_features,out_features=num_classes)

    def forward(self,x):
        with torch.no_grad():
            x = self.transformer(x)

        x = self.avgpool(x)    
        x = x.flatten(1)
        x = self.fc(x)
        return x