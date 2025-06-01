import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet18, ResNet18_Weights

from architectures.unet_parts import EfficientDown, UpMid, OutConv


class EfficientUNetWithClinicalClassification(nn.Module):
    def __init__(self, in_channels=1, n_segmentation_classes=1, num_classes=2, bilinear=False):
        super(EfficientUNetWithClinicalClassification, self).__init__()

        # Load pretrained ResNet18
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        base_model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.base_model = base_model

        # Encoder from ResNet
        self.inc = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
        self.maxpool = base_model.maxpool
        self.down1 = base_model.layer1  # 64
        self.down2 = base_model.layer2  # 128
        self.down3 = base_model.layer3  # 256
        self.down4 = base_model.layer4  # 512

        # Decoder blocks
        self.up1 = UpMid(512, 256, 256, bilinear)
        self.up2 = UpMid(256, 128, 128, bilinear)
        self.up3 = UpMid(128, 64, 64, bilinear)
        self.up4 = UpMid(64, 64, 32, bilinear)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(32, n_segmentation_classes, kernel_size=1)

        # Gated clinical fusion (restored)
        self.gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.clinical_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 128, num_classes)
        )

    def forward(self, x, clinical):
        # Encoder
        x0 = self.inc(x)
        x1 = self.maxpool(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x_seg = self.up1(x5, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x0)
        x_seg = self.final_up(x_seg)
        seg_logits = self.outc(x_seg)

        # Classification
        pooled = self.global_avg_pool(x5).view(x5.size(0), -1)  # [B, 512]

        menopause = clinical[:, 0:1]  # [B, 1]
        hospital = clinical[:, 1:2]   # [B, 1]
        gate_value = self.gate(hospital)  # [B, 1]
        gated_menopause = gate_value * menopause
        gated_clinical = torch.cat([gated_menopause, hospital], dim=1)  # [B, 2]

        clinical_embedding = self.clinical_proj(gated_clinical)  # [B, 128]
        x_cls = torch.cat((pooled, clinical_embedding), dim=1)  # [B, 640]
        cls_logits = self.classifier(x_cls)

        return seg_logits, cls_logits
class EfficientUNetWithClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes, bilinear=False):
        super(EfficientUNetWithClassification, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes
        self.num_classes = num_classification_classes
        self.bilinear = bilinear

        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        features = list(effnet.features.children())

        # Encoder
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=1),
            features[0]
        )
        self.down1 = EfficientDown([features[1]])
        self.down2 = EfficientDown([features[2]])
        self.down3 = EfficientDown([features[3]])
        self.down4 = EfficientDown([features[4]])

        self.mid = features[5]
        self.segmentation_deep = features[6]

        self.classification_conv = nn.Sequential(features[7], features[8])

        # Decoder
        self.up1 = UpMid(192, 80, 80, bilinear)
        self.up2 = UpMid(80, 40, 40, bilinear)
        self.up3 = UpMid(40, 24, 24, bilinear)
        self.up4 = UpMid(24, 32, 16, bilinear)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(16, n_segmentation_classes)

        self.global_avg_pool = effnet.avgpool
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classification_classes, bias=True)
        )

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.mid(x4)
        x6 = self.segmentation_deep(x5)

        # Segmentation decoder
        x_seg = self.up1(x6, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x0)
        x_seg = self.final_up(x_seg)
        seg_logits = self.outc(x_seg)

        # Classification head
        x_cls = self.classification_conv(x6)
        pooled = self.global_avg_pool(x_cls).view(x_cls.size(0), -1)
        class_logits = self.classification_head(pooled)

        return seg_logits, class_logits


def transfer_weights_to_clinical_model(old_model, new_model):
    """
    Transfers encoder and decoder weights from an old EfficientUNetWithClassification
    model to a new EfficientUNetWithClinicalClassification model.
    """
    # Get the state_dicts
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()

    # We'll update new_state_dict with matching keys from old_state_dict
    transferred_state_dict = {}

    for key in new_state_dict:
        if key in old_state_dict:
            if old_state_dict[key].shape == new_state_dict[key].shape:
                transferred_state_dict[key] = old_state_dict[key]
        #     else:
        #         print(
        #             f"Skipped (shape mismatch): {key} | old: {old_state_dict[key].shape}, new: {new_state_dict[key].shape}")
        # else:
        #     print(f"Skipped (not found): {key}")

    # Load the matching parameters into the new model
    new_model.load_state_dict(transferred_state_dict, strict=False)
    print("✅ Weight transfer complete (non-matching keys skipped).")

    return new_model

# class EfficientUNetWithClinicalClassification(nn.Module):
#     def __init__(self, n_channels, n_segmentation_classes, num_classification_classes=2, clinical_feature_dim=2,
#                  bilinear=False):
#         super(EfficientUNetWithClinicalClassification, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_segmentation_classes
#         self.bilinear = bilinear
#
#         effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
#         features = list(effnet.features.children())
#
#         # EfficientNet Encoder
#         self.inc = nn.Sequential(
#             nn.Conv2d(n_channels, 3, kernel_size=1),
#             features[0]
#         )
#         self.down1 = EfficientDown([features[1]])
#         self.down2 = EfficientDown([features[2]])
#         self.down3 = EfficientDown([features[3]])
#         self.down4 = EfficientDown([features[4]])
#         self.deep_blocks = nn.Sequential(features[5], features[6], features[7])  # output: 320 channels
#
#         # Save the classification expansion block (320 -> 1280)
#         self.classification_conv = features[8]  # EfficientNet's final conv block
#
#         # Segmentation decoder
#         self.up1 = UpMid(320, 40, 40, bilinear)
#         self.up2 = UpMid(40, 24, 24, bilinear)
#         self.up3 = UpMid(24, 16, 16, bilinear)
#         self.up4 = UpMid(16, 32, 32, bilinear)
#         self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.outc = OutConv(32, n_segmentation_classes)
#
#         # Classification branch
#         self.global_avg_pool = effnet.avgpool  # AdaptiveAvgPool2d(1)
#
#         self.gate = nn.Sequential(
#             nn.Linear(1, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1),
#             nn.Sigmoid()
#         )
#
#         self.clinical_proj = nn.Sequential(
#             nn.Linear(clinical_feature_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU()
#         )
#
#         self.classification_head = nn.Sequential(
#             nn.Dropout(p=0.2, inplace=True),
#             nn.Linear(1280 + 128, num_classification_classes, bias=True)
#         )
#
#     def forward(self, x, clinical_features):
#         # Encoder
#         x = self.inc(x)
#         x1 = x
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x6 = self.deep_blocks(x5)  # shape: [B, 320, H, W]
#
#         # Segmentation decoder (from 320)
#         x_seg = self.up1(x6, x4)
#         x_seg = self.up2(x_seg, x3)
#         x_seg = self.up3(x_seg, x2)
#         x_seg = self.up4(x_seg, x1)
#         x_seg = self.final_up(x_seg)
#         seg_logits = self.outc(x_seg)
#
#         # Classification branch
#         x_cls = self.classification_conv(x6)  # shape: [B, 1280, H', W']
#         pooled = self.global_avg_pool(x_cls).view(x_cls.size(0), -1)  # shape: [B, 1280]
#
#         # Clinical gating
#         menopause = clinical_features[:, 0:1]
#         hospital = clinical_features[:, 1:2]
#         gate_value = self.gate(hospital)
#         gated_menopause = gate_value * menopause
#         gated_clinical = torch.cat([gated_menopause, hospital], dim=1)
#
#         clinical_embedding = self.clinical_proj(gated_clinical)  # shape: [B, 128]
#
#         # Classification head
#         combined = torch.cat([pooled, clinical_embedding], dim=1)  # shape: [B, 1408]
#         class_logits = self.classification_head(combined)
#
#         return seg_logits, class_logits


if __name__ == "__main__":
    old_model = EfficientUNetWithClassification(1, 1, 8)
    old_model.load_state_dict(
        torch.load("models/mmotu/joint/efficientnet_joint.pt", weights_only=True, map_location=torch.device("cpu")))
    new_model = EfficientUNetWithClinicalClassification(1, 1, 1)
    new_model = transfer_weights_to_clinical_model(old_model, new_model)
