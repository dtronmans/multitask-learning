from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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