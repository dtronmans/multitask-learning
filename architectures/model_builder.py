import os

import torch
from torch import nn

from architectures.classification_only.efficientnet_only_classification import EfficientNetClinical, \
    EfficientClassificationOnly
from architectures.classification_only.unet_classification_only import UNetClassificationOnly, UNetClinical
from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClassification, \
    EfficientUNetWithClinicalClassification, transfer_weights_to_clinical_model
from architectures.mtl.unet_with_classification import UNetWithClassification, UNetWithClinicalClassification
from architectures.segmentation_only.efficientnet_only_segmentation import EfficientUNet
from architectures.unet_parts import BasicUNet
from enums import Task, Backbone


def return_model(task, backbone, denoised=False,
                 clinical=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        base_path = os.path.join("LUMCREDACTED", "mmotu")
    else:
        base_path = os.path.join("models", "mmotu")
    if task == Task.JOINT:
        if backbone == Backbone.EFFICIENTNET:
            if denoised:
                base_path = os.path.join(base_path, "joint", "efficientnet_joint_denoised.pt")
            else:
                base_path = os.path.join(base_path, "joint", "efficientnet_joint.pt")
            if clinical:
                old_model = EfficientUNetWithClassification(1, 1, 8)
                old_model.load_state_dict(
                    torch.load(base_path, weights_only=True,
                               map_location=torch.device(device)))
                old_model.to(device)
                new_model = EfficientUNetWithClinicalClassification(1, 1, 2)
                new_model = transfer_weights_to_clinical_model(old_model, new_model)
                new_model.to(device)
                return new_model
            else:
                model = EfficientUNetWithClassification(1, 1, 8)
                model.load_state_dict(torch.load(base_path, weights_only=True, map_location=device))
                model.classification_head = nn.Sequential(
                    nn.Linear(1280, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )
                model.to(device)
                return model
        elif backbone == Backbone.CLASSIC:
            if denoised:
                base_path = os.path.join(base_path, "joint", "unet_joint_denoised.pt")
            else:
                base_path = os.path.join(base_path, "joint", "unet_joint.pt")
            if clinical:
                old_model = UNetWithClassification(1, 1, 8)
                old_model.load_state_dict(
                    torch.load(base_path, weights_only=True,
                               map_location=torch.device(device)))
                old_model.to(device)
                new_model = UNetWithClinicalClassification(1, 1, 2)
                new_model = transfer_weights_to_clinical_model(old_model, new_model)
                new_model.to(device)
                return new_model
            else:
                model = UNetWithClassification(1, 1, 8)
                model.load_state_dict(torch.load(base_path, weights_only=True, map_location=device))
                model.classification_head = nn.Sequential(
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )
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
                model = EfficientNetClinical(efficientnet_model, num_classification_classes=2)
                model.classification_head = nn.Sequential(
                    nn.Linear(1408, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )
                model.to(device)
                return model
            else:
                efficientnet_model.classification_head = nn.Sequential(
                    nn.Linear(1280, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )
                efficientnet_model.to(device)
                return efficientnet_model
        elif backbone == Backbone.CLASSIC:
            unet_model = UNetClassificationOnly(1, 8)
            if denoised:
                base_path = os.path.join(base_path, "classification", "unet_classification_denoised.pt")
            else:
                base_path = os.path.join(base_path, "classification", "unet_classification.pt")
            unet_model.load_state_dict(
                torch.load(
                    base_path,
                    weights_only=True,
                    map_location=torch.device(device)))
            unet_model.to(device)
            if clinical:
                model = UNetClinical(unet_model, 2, 2)
                model.classification_head = nn.Sequential(
                    nn.Linear(1024 + 128, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )
                model.to(device)
                return model
            else:
                unet_model.classification_head = nn.Sequential(
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )
                unet_model.to(device)
                return unet_model
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
        elif backbone == Backbone.CLASSIC:
            model = BasicUNet(1, 1)
            if denoised:
                base_path = os.path.join(base_path, "segmentation", "unet_segmentation_denoised.pt")
            else:
                base_path = os.path.join(base_path, "segmentation", "unet_segmentation.pt")
            model.load_state_dict(
                torch.load(base_path, weights_only=True,
                           map_location=device)
            )
            model.to(device)
            return model
