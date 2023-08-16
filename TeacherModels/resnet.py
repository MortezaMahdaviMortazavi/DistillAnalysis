import torchvision.models as models
import torch.nn as nn
import torch

class Resnet(nn.Module):
    def __init__(self,resnet_name,num_classes):
        super().__init__()
        assert resnet_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if resnet_name == 'resnet18':
            modules = list(models.resnet18(pretrained=True).children())[:-2]
            in_features = 512
        elif resnet_name == 'resnet34':
            modules = list(models.resnet34(pretrained=True).children())[:-2]
            in_features = 2048
        elif resnet_name == 'resnet50':
            modules = list(models.resnet50(pretrained=True).children())[:-2]
            in_features = 2048
        elif resnet_name == 'resnet101':
            modules = list(models.resnet101(pretrained=True).children())[:-2]
            in_features = 2048
        elif resnet_name == 'resnet152':
            modules = list(models.resnet152(pretrained=True).children())[:-2]
            in_features = 2048

        self.convnet = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=in_features,out_features=num_classes)

    def forward(self,x):
        with torch.no_grad():
            x = self.convnet(x)
            x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x