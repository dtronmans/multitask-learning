import torch
from torch import nn
from torchvision.models import efficientnet_b0

from architectures.unet_parts import OutConv, UpMid, EfficientDown


class EfficientUNetWithClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes, bilinear=False):
        super(EfficientUNetWithClassification, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes
        self.num_classes = num_classification_classes
        self.bilinear = bilinear

        effnet = efficientnet_b0()
        features = list(effnet.features.children())

        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=1),  # Project input to 3 channels
            features[0]
        )
        self.down1 = EfficientDown([features[1]])  # output: 16 channels
        self.down2 = EfficientDown([features[2]])  # output: 24 channels
        self.down3 = EfficientDown([features[3]])  # output: 40 channels
        self.down4 = EfficientDown([features[4]])  # output: 80 channels

        self.deep_blocks = nn.Sequential(features[5], features[6], features[7])  # final: 320 channels

        # Segmentation decoder
        self.up1 = UpMid(320, 40, 40, bilinear)
        self.up2 = UpMid(40, 24, 24, bilinear)
        self.up3 = UpMid(24, 16, 16, bilinear)
        self.up4 = UpMid(16, 32, 32, bilinear)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(32, n_segmentation_classes)

        # Classification branch
        self.classification_conv_1 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.classification_conv_2 = nn.Sequential(
            nn.Conv2d(40, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classification_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classification_classes)
        )

    def forward(self, x):
        x = self.inc(x)
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.deep_blocks(x5)

        # Segmentation path
        x_seg = self.up1(x5, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        x_seg = self.final_up(x_seg)
        seg_logits = self.outc(x_seg)

        # Classification path
        x5_features = self.classification_conv_1(x5)
        x4_features = self.classification_conv_2(x4)

        x5_pooled = self.global_avg_pool(x5_features).view(x5_features.size(0), -1)
        x4_pooled = self.global_avg_pool(x4_features).view(x4_features.size(0), -1)

        classification_input = torch.cat((x5_pooled, x4_pooled), dim=1)
        class_logits = self.classification_fc(classification_input)

        return seg_logits, class_logits