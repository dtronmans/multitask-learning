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
        self.patient_list = pd.read_csv(os.path.join(root_dir, 'patient_info.csv'))
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
            study_id = row['STUDY_ID']
            clinical_info[study_id] = {
                'menopausal_status': row.get('Menopausal status', 'Unknown'),
                'malignancy': 'malignant' if row['Malignancy status'] == 1 else 'benign',
                'hospital': 0 if study_id.startswith('LUM') else 1
            }

        for rel_path in split_filenames:
            label = rel_path.split(os.sep)[0]
            filename = os.path.basename(rel_path)

            image_path = os.path.join(self.images_dir, rel_path)
            mask_path = os.path.join(self.masks_dir, rel_path)

            if self.mask_only and not os.path.exists(mask_path):
                continue

            base_id = filename.split('_')[0]
            info = clinical_info.get(base_id, {'menopausal_status': 0, 'hospital': 1})

            if info['hospital'] == "Unknown":
                print(f"Warning: Missing clinical info for {filename}")

            self.samples.append({
                'image_path': image_path,
                'mask_path': mask_path if os.path.exists(mask_path) else None,
                'label': 0 if label == "benign" else 1,
                'menopausal_status': info['menopausal_status'],
                'hospital': info['hospital'],
                'clinical': torch.tensor([info['menopausal_status'], info['hospital']], dtype=torch.float32)
            })
        self.samples = self.samples[::-1]

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
            'hospital': sample['hospital'],
            'clinical': sample['clinical']
        }


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor()
    ])
    dataset = MedicalImageDataset("../final_datasets/once_more/mtl_denoised", split="train", transform=transform,
                                  mask_only=True)
    for i in range(len(dataset)):
        print(dataset[i]['filename'])
        print(dataset[i]['menopausal_status'])
