import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from architectures.unet_parts import EfficientDown, UpMid, OutConv


class EfficientUNetWithClinicalClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes=2, clinical_feature_dim=2, bilinear=False):
        super(EfficientUNetWithClinicalClassification, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes
        self.bilinear = bilinear

        # Feature extractor backbone
        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        features = list(effnet.features.children())

        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=1),
            features[0]
        )
        self.down1 = EfficientDown([features[1]])
        self.down2 = EfficientDown([features[2]])
        self.down3 = EfficientDown([features[3]])
        self.down4 = EfficientDown([features[4]])
        self.deep_blocks = nn.Sequential(features[5], features[6], features[7])

        # Segmentation decoder (unchanged)
        self.up1 = UpMid(320, 40, 40, bilinear)
        self.up2 = UpMid(40, 24, 24, bilinear)
        self.up3 = UpMid(24, 16, 16, bilinear)
        self.up4 = UpMid(16, 32, 32, bilinear)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(32, n_segmentation_classes)

        # Classification branch
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.gate = nn.Sequential(
            nn.Linear(1, 8),  # input: hospital feature only
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Project transformed clinical features
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Final classification head w/ clinical input
        self.classification_head = nn.Sequential(
            nn.Linear(320 + 128, 128),  # 320 from CNN, 2 clinical inputs
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classification_classes)
        )

    def forward(self, x, clinical_features):
        x = self.inc(x)
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.deep_blocks(x5)

        x_seg = self.up1(x6, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        x_seg = self.final_up(x_seg)
        seg_logits = self.outc(x_seg)

        pooled = self.global_avg_pool(x6).view(x6.size(0), -1)  # shape: (B, 320)

        menopause = clinical_features[:, 0:1]  # shape [B, 1]
        hospital = clinical_features[:, 1:2]   # shape [B, 1]

        gate_value = self.gate(hospital)
        gated_menopause = gate_value * menopause
        gated_clinical = torch.cat([gated_menopause, hospital], dim=1)

        clinical_embedding = self.clinical_proj(gated_clinical)  # shape [B, 128]

        combined = torch.cat([pooled, clinical_embedding], dim=1)
        class_logits = self.classification_head(combined)

        return seg_logits, class_logits


def transfer_weights_to_clinical_model(old_model, new_model):
    """
    Transfers encoder and decoder weights from an old EfficientUNetWithClassification
    model to a new EfficientUNetWithClinicalClassification model.

    Args:
        old_model (EfficientUNetWithClassification): Pretrained model with 8-class classification.
        new_model (EfficientUNetWithClinicalClassification): New model expecting 2-class classification with clinical input.

    Returns:
        EfficientUNetWithClinicalClassification: The new model with transferred weights.
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


class EfficientUNetWithClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes, bilinear=False):
        super(EfficientUNetWithClassification, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes
        self.num_classes = num_classification_classes
        self.bilinear = bilinear

        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
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
            nn.Conv2d(80, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classification_classes)
        )

    def forward(self, x):
        x = self.inc(x)
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.deep_blocks(x5)

        # Segmentation path
        x_seg = self.up1(x6, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x1)
        x_seg = self.final_up(x_seg)
        seg_logits = self.outc(x_seg)

        pooled = self.global_avg_pool(x6).view(x6.size(0), -1)
        class_logits = self.classification_head(pooled)

        return seg_logits, class_logits


if __name__ == "__main__":
    old_model = EfficientUNetWithClassification(1, 1, 8)
    old_model.load_state_dict(
        torch.load("models/mmotu/joint/efficientnet_joint.pt", weights_only=True, map_location=torch.device("cpu")))
    new_model = EfficientUNetWithClinicalClassification(1, 1, 1)
    new_model = transfer_weights_to_clinical_model(old_model, new_model)
