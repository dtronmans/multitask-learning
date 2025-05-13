import os

import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from architectures.classification_only.efficientnet_only_classification import EfficientNetClinical
from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClassification, \
    EfficientUNetWithClinicalClassification, transfer_weights_to_clinical_model
from architectures.segmentation_only.efficientnet_only_segmentation import EfficientUNet
from enums import Task, Backbone


def return_model(task, backbone, denoised=False, clinical=False):  # here we return the models, with the clinical information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        base_path = os.path.join("/exports", "lkeb-hpc", "dzrogmans", "models", "mmotu")
    else:
        base_path = os.path.join("models", "mmotu")
    if task == Task.JOINT:
        if backbone == Backbone.EFFICIENTNET:
            if clinical:
                old_model = EfficientUNetWithClassification(1, 1, 8)
                if denoised:
                    base_path = os.path.join(base_path, "efficientnet_joint_denoised.pt")
                else:
                    base_path = os.path.join(base_path, "efficientnet_joint.pt")
                old_model.load_state_dict(
                    torch.load(base_path, weights_only=True,
                               map_location=torch.device(device)))
                new_model = EfficientUNetWithClinicalClassification(1, 1, 2)
                new_model = transfer_weights_to_clinical_model(old_model, new_model)
                new_model.to(device)
                return new_model
            else:
                if denoised:
                    base_path = os.path.join(base_path, "efficientnet_joint_denoised.pt")
                else:
                    base_path = os.path.join(base_path, "efficientnet_joint.pt")
                model = EfficientUNetWithClinicalClassification(1, 1, 8)
                model.load_state_dict(torch.load(base_path))
                model.classification_head[3] = nn.Linear(1280, 2)
    if task == Task.CLASSIFICATION:
        if backbone == Backbone.EFFICIENTNET:
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
            if denoised:
                base_path = os.path.join(base_path, "efficientnet_classification_denoised.pt")
            else:
                base_path = os.path.join(base_path, "efficientnet_classification.pt")
            efficientnet_model.load_state_dict(
                torch.load(
                    base_path,
                    weights_only=True,
                    map_location=torch.device(device)))
            efficientnet_model.to(device)
            model = EfficientNetClinical(efficientnet_model, num_classes=2)
            model.to(device)
            return model
    if task == Task.SEGMENTATION:
        if backbone == Backbone.EFFICIENTNET:
            model = EfficientUNet(1, 1)
            base_path = os.path.join("models", "mmotu", "segmentation")
            if denoised:
                base_path = os.path.join(base_path, "efficientnet_segmentation_denoised.pt")
            else:
                base_path = os.path.join(base_path, "efficientnet_segmentation.pt")
            model.load_state_dict(
                torch.load(base_path,
                           weights_only=True,
                           map_location=torch.device("cpu")))
            return model
