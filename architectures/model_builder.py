import os

import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from architectures.classification_only.efficientnet_only_classification import EfficientNetClinical, \
    EfficientClassificationOnly
from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClassification, \
    EfficientUNetWithClinicalClassification, transfer_weights_to_clinical_model
from architectures.mtl.resnet_with_classification import ResNetUNetWithClinicalClassification
from architectures.segmentation_only.efficientnet_only_segmentation import EfficientUNet
from enums import Task, Backbone


def return_model(task, backbone, denoised=False,
                 clinical=False):  # here we return the models, with the clinical information
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
                    base_path = os.path.join(base_path, "joint", "efficientnet_joint_denoised.pt")
                else:
                    base_path = os.path.join(base_path, "joint", "efficientnet_joint.pt")
                old_model.load_state_dict(
                    torch.load(base_path, weights_only=True,
                               map_location=torch.device(device)))
                new_model = EfficientUNetWithClinicalClassification(1, 1, 2)
                new_model = transfer_weights_to_clinical_model(old_model, new_model)
                new_model.to(device)
                return new_model
            else:
                if denoised:
                    base_path = os.path.join(base_path, "joint", "efficientnet_joint_denoised.pt")
                else:
                    base_path = os.path.join(base_path, "joint", "efficientnet_joint.pt")
                model = EfficientUNetWithClassification(1, 1, 8)
                model.load_state_dict(torch.load(base_path, weights_only=True, map_location=device))
                model.classification_head = nn.Sequential(
                    nn.Linear(1280, 128),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(128, 2)
                )
                model.to(device)
                return model
        elif backbone == Backbone.RESNET:
            model = ResNetUNetWithClinicalClassification(1, 1, 2)
            model.to(device)
            return model
    if task == Task.CLASSIFICATION:
        if backbone == Backbone.EFFICIENTNET:
            efficientnet_model = EfficientClassificationOnly(1, 8)
            if denoised:
                base_path = os.path.join(base_path, "classification", "efficientnet_classification_denoised.pt")
            else:
                base_path = os.path.join(base_path, "classification", "efficientnet_classification.pt")
            efficientnet_model.load_state_dict(
                torch.load(
                    base_path,
                    weights_only=True,
                    map_location=torch.device(device)))
            efficientnet_model.to(device)
            if clinical:
                model = EfficientNetClinical(efficientnet_model, num_classes=2)
                model.to(device)
                model = transfer_weights_to_clinical_model(efficientnet_model, model)
                return model
            else:
                efficientnet_model.classification_head[1] = nn.Linear(1280, 2, bias=True)
                efficientnet_model.to(device)
                return efficientnet_model
    if task == Task.SEGMENTATION:
        if backbone == Backbone.EFFICIENTNET:
            model = EfficientUNet(1, 1)
            if denoised:
                base_path = os.path.join(base_path, "segmentation", "efficientnet_segmentation_denoised.pt")
            else:
                base_path = os.path.join(base_path, "segmentation", "efficientnet_segmentation.pt")
            model.load_state_dict(
                torch.load(base_path,
                           weights_only=True,
                           map_location=device))
            model.to(device)
            return model
