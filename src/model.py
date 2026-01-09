import torch
import torch.nn as nn
import timm


class DistilledMobileNet(nn.Module):
    def __init__(self, num_classes=89):
        super().__init__()
        self.backbone = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        with torch.no_grad():
            dummy_output = self.backbone(torch.zeros(1, 3, 224, 224))
            self.num_features = dummy_output.shape[1]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits
