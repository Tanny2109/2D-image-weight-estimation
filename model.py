import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

class MultiViewNutritionModel(nn.Module):
    def __init__(self, output_dim=12, freeze_backbone=True):
        super(MultiViewNutritionModel, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.backbone.fc = nn.Identity()
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, output_dim)

    def forward(self, x):
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            x = x.view(B * V, C, H, W)
        else:
            B, C, H, W = x.shape
            V = 1
            x = x.view(B * V, C, H, W)

        feats = self.backbone(x)
        feats = feats.view(B, V, -1)
        fused_feats = feats.mean(dim=1)
        x = self.fc1(fused_feats)
        x = self.relu(x)
        return self.out(x)