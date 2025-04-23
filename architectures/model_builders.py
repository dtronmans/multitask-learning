import torch
from torch import nn
from torchvision import models

from architectures.unet_parts import BasicUNet, EfficientUNet, ResNetUNet


def classification_architecture(model, clinical):
    if model == "efficientnet":
        model = efficientnet_classification("models/mmotu/mmotu_efficientnet_b0.pt")
        if clinical:
            return EfficientNetWithClinical(model)
    elif model == "resnet":
        model = resnet18_classification("models/mmotu/mmotu_resnet18.pt")
        if clinical:
            return ResNet18WithClinical(model)
    return model


def segmentation_architecture(model):
    if model == "efficientnet":
        model = EfficientUNet(1, 1)
    elif model == "resnet":
        model = ResNetUNet(1, 1)
    elif model == "classic":
        model = BasicUNet(1, 1)
    return model


class EfficientNetWithClinical(nn.Module):
    def __init__(self, base_model, clinical_feature_dim=2, num_classes=2):
        super(EfficientNetWithClinical, self).__init__()

        image_feat_dim = base_model.classifier[1].in_features

        base_model.classifier = nn.Identity()
        self.base_model = base_model

        self.gate = nn.Sequential(
            nn.Linear(1, 8),
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

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(image_feat_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, clinical_features):
        # clinical_features: shape [B, 2] -> [menopause, hospital]
        menopause = clinical_features[:, 0:1]
        hospital = clinical_features[:, 1:2]

        gate_value = self.gate(hospital)
        gated_menopause = gate_value * menopause

        gated_clinical = torch.cat([gated_menopause, hospital], dim=1)

        clinical_embedding = self.clinical_proj(gated_clinical)
        image_features = self.base_model(x)
        x = torch.cat((image_features, clinical_embedding), dim=1)
        out = self.classifier(x)
        return out


def efficientnet_classification(pretrained_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base = models.efficientnet_b0()
    model_base.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )

    model_base.classifier[1] = nn.Linear(model_base.classifier[1].in_features, 8)
    model_base.load_state_dict(torch.load(pretrained_path, weights_only=True, map_location=device))
    model_base.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model_base.classifier[1].in_features, 2)
    )
    model_base.to(device)
    model_base.train()
    return model_base


class ResNet18WithClinical(nn.Module):
    def __init__(self, base_model, clinical_feature_dim=2, num_classes=2):
        super(ResNet18WithClinical, self).__init__()

        image_feat_dim = base_model.fc.in_features
        base_model.fc = nn.Identity()  # Remove final classification layer
        self.base_model = base_model

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

        self.classifier = nn.Sequential(
            nn.Linear(image_feat_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, clinical_features):
        menopause = clinical_features[:, 0:1]
        hospital = clinical_features[:, 1:2]

        gate_value = self.gate(hospital)
        gated_menopause = gate_value * menopause

        gated_clinical = torch.cat([gated_menopause, hospital], dim=1)
        clinical_embedding = self.clinical_proj(gated_clinical)

        image_features = self.base_model(x)
        x = torch.cat((image_features, clinical_embedding), dim=1)
        out = self.classifier(x)
        return out


def resnet18_classification(pretrained_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base = models.resnet18(pretrained=False)

    model_base.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model_base.load_state_dict(torch.load(pretrained_path, map_location=device))

    model_base.to(device)
    model_base.train()
    return model_base
