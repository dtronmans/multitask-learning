import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetClinical(nn.Module):
    def __init__(self, backbone_model, clinical_feature_dim=2, num_classes=2):
        super(EfficientNetClinical, self).__init__()
        self.backbone = backbone_model

        # Replace the input Conv2d if input is not RGB (e.g., grayscale input with 1 channel)
        original_conv = self.backbone.features[0][0]
        if original_conv.in_channels != 1:
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )

        # Global average pooling to get a single vector from CNN
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Gating mechanism for one clinical feature (e.g., hospital)
        self.gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Clinical feature embedding
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Final classification head
        self.classification_head = nn.Sequential(
            nn.Linear(1280 + 128, 128),  # 1280 from EfficientNet-B0 final features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, clinical_features):
        features = self.backbone.features(x)
        pooled = self.global_avg_pool(features).view(features.size(0), -1)  # (B, 1280)

        menopause = clinical_features[:, 0:1]  # shape [B, 1]
        hospital = clinical_features[:, 1:2]  # shape [B, 1]

        gate_value = self.gate(hospital)
        gated_menopause = gate_value * menopause
        gated_clinical = torch.cat([gated_menopause, hospital], dim=1)

        clinical_embedding = self.clinical_proj(gated_clinical)  # (B, 128)

        combined = torch.cat([pooled, clinical_embedding], dim=1)
        class_logits = self.classification_head(combined)

        return None, class_logits


if __name__ == "__main__":
    efficientnet_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    original_conv = efficientnet_model.features[0][0]
    efficientnet_model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    efficientnet_model.load_state_dict(
        torch.load("models/mmotu/classification/efficientnet_classification_intermediate.pt", weights_only=True,
                   map_location=torch.device("cpu")))

    model = EfficientNetClinical(efficientnet_model)
