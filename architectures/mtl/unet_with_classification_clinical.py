import torch

from architectures.mtl.unet_with_classification import UNetWithClassification


def transfer_mmotu_unet_with_classification(model_path):
    device = torch.device("cpu" if torch.cuda.is_available() else "cuda")
    mmotu_unet_with_classification = UNetWithClassification(1, 1, 8)
    mmotu_unet_with_classification.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    mmotu_unet_with_classification.to(device)
