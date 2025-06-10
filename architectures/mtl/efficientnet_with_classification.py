import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from architectures.unet_parts import UpMid, OutConv


class EfficientUNetWithClinicalClassification(nn.Module):
    def __init__(self, in_channels=1, n_segmentation_classes=1, num_classes=2, bilinear=False):
        super(EfficientUNetWithClinicalClassification, self).__init__()

        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        base_model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        features = list(base_model.features.children())
        self.inc = features[0]
        self.down1 = features[1]
        self.down2 = features[2]
        self.down3 = features[3]
        self.down4 = features[4]
        self.mid = features[5]
        self.segmentation_deep = features[6]

        self.classification_conv = nn.Sequential(features[7], features[8])  # x7

        self.up1 = UpMid(1280, 80, 80, bilinear)
        self.up2 = UpMid(80, 40, 40, bilinear)
        self.up3 = UpMid(40, 24, 24, bilinear)
        self.up4 = UpMid(24, 32, 16, bilinear)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(16, n_segmentation_classes, kernel_size=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # To pool x6 as well

        self.clinical_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Adjusted input size: x6(192) + x7(1280) + clinical(128)
        self.classifier = nn.Sequential(
            nn.Linear(192 + 1280 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, clinical):
        # Encoder path
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.mid(x4)
        x6 = self.segmentation_deep(x5)
        x7 = self.classification_conv(x6)

        # Decoder path
        x_seg = self.up1(x7, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x0)
        x_seg = self.final_up(x_seg)
        seg_logits = self.outc(x_seg)

        # Classification
        pooled_x6 = self.global_avg_pool(x6).view(x6.size(0), -1)
        pooled_x7 = self.global_avg_pool(x7).view(x7.size(0), -1)

        menopause = clinical[:, 0:1]
        hospital = clinical[:, 1:2]
        gated_clinical = torch.cat([menopause, hospital], dim=1)
        clinical_embedding = self.clinical_proj(gated_clinical)

        x_cls = torch.cat((pooled_x6, pooled_x7, clinical_embedding), dim=1)
        cls_logits = self.classifier(x_cls)

        return seg_logits, cls_logits


class EfficientUNetWithClassification(nn.Module):
    def __init__(self, n_channels, n_segmentation_classes, num_classification_classes, bilinear=False):
        super(EfficientUNetWithClassification, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_segmentation_classes
        self.num_classes = num_classification_classes
        self.bilinear = bilinear

        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        base_model.features[0][0] = nn.Conv2d(
            n_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        features = list(base_model.features.children())
        self.inc = features[0]
        self.down1 = features[1]
        self.down2 = features[2]
        self.down3 = features[3]
        self.down4 = features[4]

        self.mid = features[5]
        self.segmentation_deep = features[6]

        self.classification_conv = nn.Sequential(features[7], features[8])

        # Decoder
        self.up1 = UpMid(1280, 80, 80, bilinear)
        self.up2 = UpMid(80, 40, 40, bilinear)
        self.up3 = UpMid(40, 24, 24, bilinear)
        self.up4 = UpMid(24, 32, 16, bilinear)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(16, n_segmentation_classes)

        self.global_avg_pool = base_model.avgpool
        self.classification_head = nn.Sequential(
            nn.Linear(in_features=1280, out_features=256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classification_classes, bias=True)
        )

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.mid(x4)
        x6 = self.segmentation_deep(x5)
        x7 = self.classification_conv(x6)

        # Segmentation decoder
        x_seg = self.up1(x7, x4)
        x_seg = self.up2(x_seg, x3)
        x_seg = self.up3(x_seg, x2)
        x_seg = self.up4(x_seg, x0)
        x_seg = self.final_up(x_seg)
        seg_logits = self.outc(x_seg)

        # Classification head
        pooled = self.global_avg_pool(x7).view(x7.size(0), -1)
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
            # else:
            #     print(
            #         f"Skipped (shape mismatch): {key} | old: {old_state_dict[key].shape}, new: {new_state_dict[key].shape}")
        # else:
        #     print(f"Skipped (not found): {key}")
    #
    # Load the matching parameters into the new model
    new_model.load_state_dict(transferred_state_dict, strict=False)
    print("✅ Weight transfer complete (non-matching keys skipped).")

    return new_model


if __name__ == "__main__":
    old_model = EfficientUNetWithClassification(1, 1, 8)
    old_model.load_state_dict(
        torch.load("models/mmotu/joint/efficientnet_joint.pt", weights_only=True, map_location=torch.device("cpu")))
    new_model = EfficientUNetWithClinicalClassification(1, 1, 1)
    new_model = transfer_weights_to_clinical_model(old_model, new_model)
