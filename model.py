import torch
import torch.nn as nn
from timm import create_model

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNeXtModel, self).__init__()
        self.backbone = create_model('convnext_base', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features  # 1024 for ConvNeXt-B
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        class_out = self.classifier(features)
        return class_out

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_stages(self, stages=[2, 3]):
        for name, param in self.backbone.named_parameters():
            for stage in stages:
                if f"stages.{stage}" in name:
                    param.requires_grad = True

if __name__ == "__main__":
    model = ConvNeXtModel()
    print(model)