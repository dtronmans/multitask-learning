import os
import random

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_only=False):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.patient_list = pd.read_csv(os.path.join(root_dir, 'lumc_rdgg_attributes_filtered.csv'))
        self.transform = transform

        self.samples = []

        # Create a mapping from Study ID to clinical information
        clinical_info = {}
        for _, row in self.patient_list.iterrows():
            study_id = row['Study ID']
            clinical_info[study_id] = {
                'menopausal_status': row.get('Menopausal status', 'Unknown'),
                'malignancy': 'malignant' if row['Malignancy status'] == 1 else 'benign',
                'hospital': 'LUMC' if study_id.startswith('LUMC') else 'RDG'
            }

        for label in ['benign', 'malignant']:
            image_folder = os.path.join(self.images_dir, label)
            mask_folder = os.path.join(self.masks_dir, label)

            for filename in os.listdir(image_folder):
                if filename.lower().endswith(('png', 'jpg', 'jpeg', '.tif')):
                    image_path = os.path.join(image_folder, filename)
                    mask_path = os.path.join(mask_folder, filename)
                    has_mask = os.path.exists(mask_path)
                    if not has_mask and mask_only:
                        continue

                    base_id = filename.split('_')[0]  # e.g., LUMC12345
                    info = clinical_info.get(base_id, {'menopausal_status': 'Unknown', 'hospital': 'Unknown'})

                    self.samples.append({
                        'image_path': image_path,
                        'mask_path': mask_path if has_mask else None,
                        'label': label,
                        'menopausal_status': info['menopausal_status'],
                        'hospital': info['hospital']
                    })

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('L')

        if sample['mask_path']:
            mask = Image.open(sample['mask_path']).convert('L')
            mask_transform = transforms.Compose([transforms.Resize((336, 544)), transforms.ToTensor()])
            mask = mask_transform(mask)
        else:
            mask = torch.zeros((1, 336, 544))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'mask': mask,
            'label': 0 if sample['label'] == "benign" else 1,
            'menopausal_status': 1 if sample['menopausal_status'] == 1 else 0,
            'hospital': 1 if sample['hospital'] == "RDG" else 0
        }

    def display(self, idx):
        sample = self.__getitem__(idx)
        image = transforms.ToPILImage()(sample['image'].squeeze(0)) if isinstance(sample['image'], torch.Tensor) else sample[
            'image']
        mask = transforms.ToPILImage()(sample['mask'].squeeze(0)) if sample['mask'] is not None and isinstance(sample['mask'],
                                                                                                    torch.Tensor) else \
        sample['mask']

        fig, ax = plt.subplots(1, 2 if mask else 1, figsize=(12, 6))

        title_text = f"({sample['label']})\nHospital: {sample['hospital']}, Menopausal Status: {sample['menopausal_status']}"

        if mask:
            ax[0].imshow(image, cmap='gray')
            ax[0].set_title(title_text)
            ax[0].axis('off')

            ax[1].imshow(image, cmap='gray')
            ax[1].imshow(mask, cmap='gray', alpha=0.3)
            ax[1].set_title("With Mask Overlay")
            ax[1].axis('off')
        else:
            ax.imshow(image)
            ax.set_title(title_text)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor()
    ])
    dataset = MedicalImageDataset("../final_datasets/lumc_rdg_final", transform=transform)
    for i in range(len(dataset)):
        dataset.display(i)
