from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from architectures.unet_parts import EfficientDown, UpMid, OutConv


class EfficientUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(EfficientUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        features = list(effnet.features.children())

        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=1),
            features[0]
        )
        self.down1 = EfficientDown([features[1]])
        self.down2 = EfficientDown([features[2]])
        self.down3 = EfficientDown([features[3]])
        factor = 2 if bilinear else 1
        self.down4 = EfficientDown([features[4]])

        self.deep_blocks = nn.Sequential(
            features[5],
            features[6],
            features[7],
        )

        self.up1 = UpMid(320, 40, 40, bilinear)  # 80 -> 40, skip 40 → 80 → 40
        self.up2 = UpMid(40, 24, 24, bilinear)  # 40 -> 20, skip 24 → 44 → 24
        self.up3 = UpMid(24, 16, 16, bilinear)  # 24 -> 12, skip 16 → 28 → 16
        self.up4 = UpMid(16, 32, 32, bilinear)  # 16 -> 8, skip 32 → 40 → 32
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):  # original x shape: 1 x 336 x 544
        x = self.inc(x)  # x shape now: 32 x 168 x 272
        x1 = x  # x1 shape now: 32 x 168 x 272
        x2 = self.down1(x1)  # x2 shape now: 16 x 168 x 272
        x3 = self.down2(x2)  # x3 shape now: 24 x 84 x 136
        x4 = self.down3(x3)  # x4 shape now: 40 x 42 x 68
        x5 = self.down4(x4)  # x5 shape now: 80 x 21 x 34

        x6 = self.deep_blocks(x5)

        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final_up(x)
        logits = self.outc(x)
        return logits, None
