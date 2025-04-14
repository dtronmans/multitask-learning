import os
import random

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mask_only=False):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.patient_list = pd.read_csv(os.path.join(root_dir, 'lumc_rdgg_attributes_filtered.csv'))
        self.transform = transform
        self.mask_only = mask_only

        self.samples = []

        # Load filenames for the current split (train.txt, val.txt, test.txt)
        split_file = os.path.join(root_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            split_filenames = [line.strip().replace('\\', os.sep) for line in f]

        # Create a mapping from Study ID to clinical info
        clinical_info = {}
        for _, row in self.patient_list.iterrows():
            study_id = row['Study ID']
            clinical_info[study_id] = {
                'menopausal_status': row.get('Menopausal status', 'Unknown'),
                'malignancy': 'malignant' if row['Malignancy status'] == 1 else 'benign',
                'hospital': 'LUMC' if study_id.startswith('LUM') else 'RDG'
            }

        for rel_path in split_filenames:
            label = rel_path.split(os.sep)[0]
            filename = os.path.basename(rel_path)

            image_path = os.path.join(self.images_dir, rel_path)
            mask_path = os.path.join(self.masks_dir, rel_path)

            if self.mask_only and not os.path.exists(mask_path):
                continue

            base_id = filename.split('_')[0]
            info = clinical_info.get(base_id, {'menopausal_status': 'Unknown', 'hospital': 'Unknown'})

            if info['hospital'] == "Unknown":
                print(f"Warning: Missing clinical info for {filename}")

            self.samples.append({
                'image_path': image_path,
                'mask_path': mask_path if os.path.exists(mask_path) else None,
                'label': label,
                'menopausal_status': info['menopausal_status'],
                'hospital': info['hospital']
            })

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
            'label': sample['label'],
            'menopausal_status': sample['menopausal_status'],
            'hospital': sample['hospital']
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
