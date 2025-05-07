import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from architectures.classification_only.efficientnet_only_classification import EfficientNetClinical
from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClassification, \
    EfficientUNetWithClinicalClassification, transfer_weights_to_clinical_model
from architectures.segmentation_only.efficientnet_only_segmentation import EfficientUNet
from enums import Task, Backbone


def return_model(task, backbone):  # here we return the models, with the clinical information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if task == Task.JOINT:
        if backbone == Backbone.EFFICIENTNET:
            old_model = EfficientUNetWithClassification(1, 1, 8)
            old_model.load_state_dict(
                torch.load("models/mmotu/joint/efficientnet_joint.pt", weights_only=True,
                           map_location=torch.device(device)))
            new_model = EfficientUNetWithClinicalClassification(1, 1, 2)
            new_model = transfer_weights_to_clinical_model(old_model, new_model)
            new_model.to(device)
            return new_model
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
            efficientnet_model.load_state_dict(
                torch.load("models/mmotu/classification/efficientnet_classification_intermediate.pt", weights_only=True,
                           map_location=torch.device(device)))
            efficientnet_model.to(device)
            model = EfficientNetClinical(efficientnet_model, num_classes=2)
            model.to(device)
            return model
    if task == Task.SEGMENTATION:
        if backbone == Backbone.EFFICIENTNET:
            model = EfficientUNet(1, 1)
            model.load_state_dict(
                torch.load("models/mmotu/segmentation/efficientnet_unet_intermediate.pt", weights_only=True,
                           map_location=torch.device("cpu")))
            return model
