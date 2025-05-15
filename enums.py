from enum import Enum

class Backbone(Enum):
    CLASSIC = "classic"
    EFFICIENTNET = "efficientnet"
    RESNET = "resnet"

class Task(Enum):
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    JOINT = "joint"