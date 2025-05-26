import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from architectures.unet_parts import EfficientDown, UpMid, OutConv


class EfficientUNetWithClinicalClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes, bilinear=False):
        super(EfficientUNetWithClinicalClassification, self).__init__()
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
        self.up1 = UpMid(192, 80, 80, bilinear)  # From x6 (7×7, 192) and x4 (14×14, 80)
        self.up2 = UpMid(80, 40, 40, bilinear)  # 14×14 → 28×28
        self.up3 = UpMid(40, 24, 24, bilinear)  # 28×28 → 56×56
        self.up4 = UpMid(24, 32, 16, bilinear)  # 56×56 → 112×112
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 112×112 → 224×224
        self.outc = OutConv(16, n_segmentation_classes)

        self.global_avg_pool = effnet.avgpool
        self.clinical_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Clinical embedding after gating
        self.clinical_embedding = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(1280 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x, clinical):
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

        x_cls = self.classification_conv(x6)
        pooled = self.global_avg_pool(x_cls).view(x_cls.size(0), -1)  # (B, 1280)
        menopausal = clinical[:, 0:1]  # (B, 1)
        center_type = clinical[:, 1:2]  # (B, 1)
        gate = 1 - self.clinical_gate(center_type)  # (B, 1), so higher center_type → lower influence
        modulated_menopausal = menopausal * gate  # (B, 1)

        clinical_input = torch.cat([modulated_menopausal, center_type], dim=1)  # (B, 2)
        clinical_emb = self.clinical_embedding(clinical_input)  # (B, 128)
        combined = torch.cat([pooled, clinical_emb], dim=1)  # (B, 1408)
        class_logits = self.classification_head(combined)

        return seg_logits, class_logits


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
        self.up1 = UpMid(192, 80, 80, bilinear)  # From x6 (7×7, 192) and x4 (14×14, 80)
        self.up2 = UpMid(80, 40, 40, bilinear)  # 14×14 → 28×28
        self.up3 = UpMid(40, 24, 24, bilinear)  # 28×28 → 56×56
        self.up4 = UpMid(24, 32, 16, bilinear)  # 56×56 → 112×112
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 112×112 → 224×224
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
            else:
                print(
                    f"Skipped (shape mismatch): {key} | old: {old_state_dict[key].shape}, new: {new_state_dict[key].shape}")
        else:
            print(f"Skipped (not found): {key}")

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
