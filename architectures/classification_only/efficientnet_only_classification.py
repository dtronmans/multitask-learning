import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetClinical(nn.Module):
    def __init__(self, backbone_model, clinical_feature_dim=2, num_classification_classes=2):
        super(EfficientNetClinical, self).__init__()
        self.backbone = backbone_model

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
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(256, num_classification_classes)
        )

    def forward(self, x, clinical_features):
        x = self.backbone.input_conv(x)
        features = self.backbone.encoder(x)
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


class EfficientClassificationOnly(nn.Module):
    def __init__(self, n_channels, num_classification_classes):
        super().__init__()
        self.n_channels = n_channels
        self.num_classes = num_classification_classes

        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        if n_channels != 3:
            effnet.features[0][0] = nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.encoder = effnet.features

        self.global_avg_pool = effnet.avgpool
        self.classification_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classification_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classification_head(x)
        return None, logits


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
