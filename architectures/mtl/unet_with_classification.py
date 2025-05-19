import torch
from torch import nn

from architectures.unet_parts import DoubleConv, Down, Up, OutConv


class UNetWithClinicalClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes, bilinear=False):
        super(UNetWithClinicalClassification, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes
        self.num_classes = num_classification_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_segmentation_classes)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.clinical_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.clinical_embedding = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear((1024 // factor) + 128, num_classification_classes)
        )

    def forward(self, x, clinical):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_seg = self.up1(x5, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        seg_logits = self.outc(x_seg)

        pooled = self.global_avg_pool(x5).view(x5.size(0), -1)

        menopausal = clinical[:, 0:1]
        center_type = clinical[:, 1:2]
        gate = 1 - self.clinical_gate(center_type)
        modulated_menopausal = menopausal * gate
        clinical_input = torch.cat([modulated_menopausal, center_type], dim=1)
        clinical_emb = self.clinical_embedding(clinical_input)

        combined = torch.cat([pooled, clinical_emb], dim=1)
        class_logits = self.classification_head(combined)

        return seg_logits, class_logits


class UNetWithClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes, bilinear=False):
        super(UNetWithClassification, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes
        self.num_classes = num_classification_classes
        self.bilinear = bilinear

        # Segmentation encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Segmentation decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_segmentation_classes)

        # Classification head (aligned with EfficientNet)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024 // factor, num_classification_classes)
        )

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Segmentation decoder path
        x_seg = self.up1(x5, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        seg_logits = self.outc(x_seg)

        # Classification head (based only on x5, like EfficientNet)
        pooled = self.global_avg_pool(x5).view(x5.size(0), -1)
        class_logits = self.classification_head(pooled)

        return seg_logits, class_logits
