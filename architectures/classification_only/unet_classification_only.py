import torch
from torch import nn

from architectures.unet_parts import DoubleConv, Down


class UNetClinical(nn.Module):
    def __init__(self, backbone_model, clinical_feature_dim=2, num_classification_classes=2):
        super(UNetClinical, self).__init__()
        self.backbone = backbone_model

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(1024 + 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(256, num_classification_classes)
        )

    def forward(self, x, clinical_features):
        x1 = self.backbone.inc(x)
        x2 = self.backbone.down1(x1)
        x3 = self.backbone.down2(x2)
        x4 = self.backbone.down3(x3)
        features = self.backbone.down4(x4)
        pooled = self.global_avg_pool(features).view(features.size(0), -1)  # (B, 1280)

        menopause = clinical_features[:, 0:1]  # shape [B, 1]
        hospital = clinical_features[:, 1:2]  # shape [B, 1]

        gated_clinical = torch.cat([menopause, hospital], dim=1)

        clinical_embedding = self.clinical_proj(gated_clinical)  # (B, 128)

        combined = torch.cat([pooled, clinical_embedding], dim=1)
        class_logits = self.classification_head(combined)

        return None, class_logits


class UNetClassificationOnly(nn.Module):
    def __init__(self, n_channels, num_classification_classes, bilinear=False):
        super(UNetClassificationOnly, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classification_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Classification head (aligned with EfficientNet and updated UNetWithClassification)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024 // factor, num_classification_classes)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        pooled = self.global_avg_pool(x5).view(x5.size(0), -1)
        class_logits = self.classification_head(pooled)
        return None, class_logits
