import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from architectures.unet_parts import BasicUNet
from dataset import MedicalImageDataset


def save_cropped_segmented_images(model, dataloader, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    malignant_dir = os.path.join(destination_folder, 'malignant')
    benign_dir = os.path.join(destination_folder, 'benign')
    os.makedirs(malignant_dir, exist_ok=True)
    os.makedirs(benign_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            image, mask, label, filename = batch['image'], batch['mask'], batch['label'], batch['filename']
            output = model(image)

            pred_mask = torch.sigmoid(output)
            pred_mask = (pred_mask > 0.5).float()

            image_np = image.squeeze().numpy()
            mask_np = pred_mask.squeeze().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            mask_np = (mask_np * 255).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue  # Skip if no contours found

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add 20% padding
            pad_x = int(0.1 * w)
            pad_y = int(0.1 * h)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + w + pad_x, image_np.shape[1])
            y2 = min(y + h + pad_y, image_np.shape[0])

            cropped = image_np[y1:y2, x1:x2]

            # Make it square by padding with black pixels
            height, width = cropped.shape
            size = max(height, width)
            square = np.zeros((size, size), dtype=np.uint8)
            y_offset = (size - height) // 2
            x_offset = (size - width) // 2
            square[y_offset:y_offset+height, x_offset:x_offset+width] = cropped

            # Determine save path
            label_value = int(label.item())
            subfolder = 'malignant' if label_value == 1 else 'benign'
            save_path = os.path.join(destination_folder, subfolder, os.path.basename(filename[0]))
            square = square.astype(np.uint8)
            img_pil = Image.fromarray(square, mode='L')
            img_pil.save(save_path)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    file_path = "../final_datasets/once_more/mtl_denoised/mtl_denoised"
    model_path = "models/hospital/unet_not_normalized_denoised.pt"
    destination_folder = "mtl_cropped"

    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor()
    ])

    dataset = MedicalImageDataset(file_path, split="test", transform=transform, mask_only=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = BasicUNet(1, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    save_cropped_segmented_images(model, dataloader, destination_folder)