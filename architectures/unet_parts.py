import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from torchvision.models import resnet18


class BasicUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(BasicUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
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
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class EfficientUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(EfficientUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load pretrained EfficientNetB0
        effnet = efficientnet_b0()
        features = list(effnet.features.children())

        # We'll use first 4 downsampling blocks
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=1),  # project input to 3 channels if needed
            features[0]
        )
        self.down1 = EfficientDown([features[1]])  # 112 -> 56
        self.down2 = EfficientDown([features[2]])  # 56 -> 28
        self.down3 = EfficientDown([features[3]])  # 28 -> 14
        factor = 2 if bilinear else 1
        self.down4 = EfficientDown([features[4]])  # 14 -> 7

        # Bottleneck and upsample path remains the same
        self.up1 = Up(112, 80 // factor, bilinear)  # channels need to match actual output
        self.up2 = Up(80, 40 // factor, bilinear)
        self.up3 = Up(40, 24 // factor, bilinear)
        self.up4 = Up(24, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ResNetUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ResNetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load pretrained resnet18
        base_model = resnet18()

        self.input_conv = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=1),
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
        )
        self.maxpool = base_model.maxpool

        self.down1 = ResNetDown(base_model.layer1)
        self.down2 = ResNetDown(base_model.layer2)
        self.down3 = ResNetDown(base_model.layer3)
        factor = 2 if bilinear else 1
        self.down4 = ResNetDown(base_model.layer4)

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.input_conv(x)
        x1 = self.maxpool(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class EfficientDown(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResNetDown(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpResidual(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x) + self.residual(x1)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
