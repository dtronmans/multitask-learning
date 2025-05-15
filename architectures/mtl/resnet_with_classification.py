import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from architectures.unet_parts import UpMid, OutConv


class ResNetUNetWithClinicalClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes=2, clinical_feature_dim=2,
                 bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes

        # Load pretrained ResNet18
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.input_layer = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=1),  # adapt to RGB
        )

        self.encoder1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)  # -> 64
        self.pool1 = base_model.maxpool
        self.encoder2 = base_model.layer1  # -> 64
        self.encoder3 = base_model.layer2  # -> 128
        self.encoder4 = base_model.layer3  # -> 256
        self.encoder5 = base_model.layer4  # -> 512

        # Segmentation decoder
        self.up1 = UpMid(512, 256, 256, bilinear)
        self.up2 = UpMid(256, 128, 128, bilinear)
        self.up3 = UpMid(128, 64, 64, bilinear)
        self.up4 = UpMid(64, 64, 32, bilinear)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(32, n_segmentation_classes)

        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(512 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classification_classes)
        )

    def forward(self, x, clinical_features):
        x = self.input_layer(x)
        x1 = self.encoder1(x)  # -> 64
        x2 = self.encoder2(self.pool1(x1))  # -> 64
        x3 = self.encoder3(x2)  # -> 128
        x4 = self.encoder4(x3)  # -> 256
        x5 = self.encoder5(x4)  # -> 512

        # Segmentation decoder
        x_seg = self.up1(x5, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        x_seg = self.final_up(x_seg)
        seg_logits = self.outc(x_seg)

        # Classification
        pooled = self.global_avg_pool(x5).view(x5.size(0), -1)  # shape: [B, 512]

        # Clinical gating
        menopause = clinical_features[:, 0:1]
        hospital = clinical_features[:, 1:2]
        gate_value = self.gate(hospital)
        gated_menopause = gate_value * menopause
        gated_clinical = torch.cat([gated_menopause, hospital], dim=1)

        clinical_embedding = self.clinical_proj(gated_clinical)  # shape: [B, 128]

        combined = torch.cat([pooled, clinical_embedding], dim=1)  # shape: [B, 640]
        class_logits = self.classification_head(combined)

        return seg_logits, class_logits
