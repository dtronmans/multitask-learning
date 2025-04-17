import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from architectures.unet_parts import BasicUNet
from dataset import MedicalImageDataset


def infer(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            image = batch['image']  # shape: [1, 1, H, W]
            output = model(image)

            # Apply sigmoid and threshold to get binary mask
            pred_mask = torch.sigmoid(output)
            pred_mask = (pred_mask > 0.5).float()

            # Convert tensors to numpy for visualization
            image_np = image.squeeze().numpy()
            mask_np = pred_mask.squeeze().numpy()

            # Normalize image for display
            image_np = (image_np * 0.5) + 0.5

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Input Image")
            plt.imshow(image_np, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Predicted Mask Overlay")
            plt.imshow(image_np, cmap='gray')
            plt.imshow(mask_np, cmap='jet', alpha=0.5)
            plt.axis('off')

            plt.show()


if __name__ == "__main__":
    file_path = "../final_datasets/once_more/mtl_final"
    model_path = "models/mmotu/mmotu_unet_grayscale.pt"
    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MedicalImageDataset(file_path, transform=transform)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = BasicUNet(1, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    infer(model, dataloader)
