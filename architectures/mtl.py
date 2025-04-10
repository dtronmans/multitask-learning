import torch
import torch.nn as nn

from architectures.unet_parts import DoubleConv, Down, Up, OutConv


class MTLNet(nn.Module):

    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes, bilinear=False):
        super(MTLNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes
        self.num_classes = num_classification_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_segmentation_classes))

        # Image feature extractor before classification
        self.classification_conv_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.classification_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classification_conv_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classification_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classification_classes)
        )

    def forward(self, x):
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

        x5_features = self.classification_conv_1(x5)  # High-level deep features
        x4_features = self.classification_conv_2(x4)  # Mid-level features

        x5_pooled = self.global_avg_pool(x5_features).view(x5_features.size(0), -1)
        x4_pooled = self.global_avg_pool(x4_features).view(x4_features.size(0), -1)

        classification_input = torch.cat((x5_pooled, x4_pooled), dim=1)

        class_logits = self.classification_fc(classification_input)

        return seg_logits, class_logits