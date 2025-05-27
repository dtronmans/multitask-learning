import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

from paired_transform import DefaultPairedTransform, PairedTransform


class MultimodalMMOTUDataset(Dataset):
    def __init__(self, dataset_path, phase="train", paired_transform=None):
        self.dataset_path = dataset_path
        self.paired_transform = paired_transform or DefaultPairedTransform()
        self.images_dir = os.path.join(dataset_path, "images")
        self.masks_dir = os.path.join(dataset_path, "annotations")
        self.num_classes = 8

        if phase == 'train':
            data_file = 'train_cls.txt'
        elif phase == 'val':
            data_file = 'val_cls.txt'
        elif phase == 'test':
            data_file = 'test_cls.txt'
        else:
            raise ValueError("Invalid phase specified. Choose 'train', 'val', or 'test'.")

        self.data = []
        with open(os.path.join(dataset_path, data_file), 'r') as file:
            for line in file:
                filename, cls = line.strip().split()
                if os.path.exists(os.path.join(self.images_dir, filename)) and filename != "3.JPG":
                    cls = int(cls)
                    self.data.append((filename, cls))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, label = self.data[index]
        img_path = os.path.join(self.images_dir, filename)
        mask_filename = os.path.splitext(filename)[0] + ".PNG"
        mask_path = os.path.join(self.masks_dir, mask_filename)

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L') if os.path.exists(mask_path) else Image.new('L', image.size)

        if self.paired_transform:
            image, mask = self.paired_transform(image, mask)

        return image, torch.tensor(label, dtype=torch.long), mask


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
            split_filenames = [line.strip().replace('\\', '/') for line in f]

        # Create a mapping from Study ID to clinical info
        clinical_info = {}
        for _, row in self.patient_list.iterrows():
            study_id = row['STUDY_ID']
            clinical_info[study_id] = {
                'menopausal_status': row.get('Menopausal status', 'Unknown'),
                'malignancy': 'malignant' if row['Malignancy status'] == 1 else 'benign',
                'hospital': 1 if study_id.lower().startswith('lum') else 0
            }

        for rel_path in split_filenames:
            label = rel_path.split(os.sep)[0]
            filename = os.path.basename(rel_path)

            image_path = os.path.join(self.images_dir, rel_path)
            mask_path = os.path.join(self.masks_dir, rel_path)
            alternate_mask_path = os.path.join(self.masks_dir, rel_path.replace("tif", "png"))

            if self.mask_only and not os.path.exists(mask_path) and not os.path.exists(alternate_mask_path):
                continue

            base_id = filename.split('_')[0]
            info = clinical_info.get(base_id, {'menopausal_status': 0, 'hospital': 1})

            if info['hospital'] == "Unknown":
                print(f"Warning: Missing clinical info for {filename}")

            if os.path.exists(alternate_mask_path): # here, we prioritize png (our) masks
                mp = alternate_mask_path
            elif os.path.exists(mask_path):
                mp = mask_path
            else:
                mp = None

            self.samples.append({
                'image_path': image_path,
                'mask_path': mp,
                'label': 0 if label.startswith("benign") else 1,
                'menopausal_status': torch.tensor(info['menopausal_status'], dtype=torch.float32),
                'hospital': torch.tensor(info['hospital'], dtype=torch.float32),
                'clinical': torch.tensor([info['menopausal_status'], info['hospital']], dtype=torch.float32)
            })
        # self.samples = self.samples[::-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('L')

        if sample['mask_path']:
            mask = Image.open(sample['mask_path']).convert('L')
        else:
            mask = Image.new('L', (544, 336))  # Create empty black mask if not available

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Default deterministic transform
            image = transforms.Resize((336, 544))(image)
            image = transforms.ToTensor()(image)
            mask = transforms.Resize((336, 544))(mask)
            mask = transforms.ToTensor()(mask)

        return {
            'image': image,
            'mask': mask,
            'label': sample['label'],
            'menopausal_status': sample['menopausal_status'],
            'hospital': sample['hospital'],
            'clinical': sample['clinical'],
            'image_path': sample['image_path']
        }

    def display(self, idx):
        sample = self[idx]
        image = sample['image'].squeeze().numpy()
        mask = sample['mask'].squeeze().numpy()
        label = "benign" if sample['label'] == 0 else "malignant"

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(label, fontsize=14)

        # Original image
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Image with semi-transparent mask overlay
        axs[1].imshow(image, cmap='gray')
        axs[1].imshow(mask, cmap='Reds', alpha=0.2)  # Alpha controls transparency
        axs[1].set_title("Image with Mask Overlay")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for idx in range(len(self.samples)):
            sample = self[idx]
            image = sample['image'].squeeze().numpy()
            mask = sample['mask'].squeeze().numpy()
            filename = os.path.basename(self.samples[idx]['image_path'])
            filename_without_ext = os.path.splitext(filename)[0]

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Filename: {filename}", fontsize=14)

            # Original image
            axs[0].imshow(image, cmap='gray')
            axs[0].set_title("Original Image")
            axs[0].axis('off')

            # Image with semi-transparent mask overlay
            axs[1].imshow(image, cmap='gray')
            axs[1].imshow(mask, cmap='Reds', alpha=0.2)
            axs[1].set_title("Image with Mask Overlay")
            axs[1].axis('off')

            plt.tight_layout()

            # Save the figure
            save_path = os.path.join(output_dir, f"{filename_without_ext}_overlay.png")
            plt.savefig(save_path)
            plt.close(fig)  # Very important to prevent memory leaks when saving lots of images


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor()
    ])
    dataset = MedicalImageDataset("../final_datasets/once_more/mtl_final", split="train", transform=PairedTransform(),
                                  mask_only=False)
    for i in range(len(dataset)):
        dataset.display(i)
