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
            image, mask, label, filename, clinical = batch['image'], batch['mask'], batch['label'], batch['filename'], batch['clinical']  # shape: [1, 1, H, W]
            # output = model(image)

            # Apply sigmoid and threshold to get binary mask
            # pred_mask = torch.sigmoid(output)
            # pred_mask = (pred_mask > 0.5).float()

            # Convert tensors to numpy for visualization
            image_np = image.squeeze().numpy()
            # mask_np = pred_mask.squeeze().numpy()
            mask_original = mask.squeeze().numpy()

            # Normalize image for display
            image_np = (image_np * 0.5) + 0.5

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Input Image")
            plt.imshow(image_np, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Mask Overlay")
            plt.imshow(image_np, cmap='gray')
            plt.imshow(mask_original, cmap='jet', alpha=0.5)
            plt.axis('off')

            plt.show()


if __name__ == "__main__":
    file_path = "../final_datasets/once_more/mtl_denoised"
    model_path = "models/hospital/unet_not_normalized_denoised_2.pt"
    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor()
    ])

    dataset = MedicalImageDataset(file_path, transform=transform, mask_only=True)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = BasicUNet(1, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    infer(model, dataloader)
