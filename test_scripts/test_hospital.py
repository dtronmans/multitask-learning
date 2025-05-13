import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClinicalClassification
from dataset import MedicalImageDataset
from enums import Backbone, Task
from train_scripts.mtl_training import return_model


def test_model(model, dataloader, task, device, clinical, threshold=0.3):  # <-- Add threshold parameter
    model.to(device)
    model.eval()
    correct_cls = 0
    total_cls = 0
    iou_scores = []

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=True):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)
            clinical_info = batch['clinical'].to(device)
            masks = (masks > 0).float()

            if clinical:
                predicted_seg, predicted_cls = model(images, clinical_info)
            else:
                predicted_seg, predicted_cls = model(images)

            if task in [Task.CLASSIFICATION, Task.JOINT]:
                # Assuming predicted_cls is of shape (batch_size, 2)
                probs = torch.softmax(predicted_cls, dim=1)[:, 1]  # probability of class 1
                preds = (probs >= threshold).long()  # Apply threshold

                correct_cls += (preds == labels).sum().item()
                total_cls += labels.size(0)

                true_positive += ((preds == 1) & (labels == 1)).sum().item()
                true_negative += ((preds == 0) & (labels == 0)).sum().item()
                false_positive += ((preds == 1) & (labels == 0)).sum().item()
                false_negative += ((preds == 0) & (labels == 1)).sum().item()

            if task in [Task.SEGMENTATION, Task.JOINT]:
                pred_mask = torch.sigmoid(predicted_seg) > 0.5
                pred_mask = pred_mask.float()
                intersection = (pred_mask * masks).sum()
                union = (pred_mask + masks).clamp(0, 1).sum()
                iou = intersection / union if union != 0 else torch.tensor(1.0)
                iou_scores.append(iou.item())

    if task in [Task.CLASSIFICATION, Task.JOINT]:
        acc = 100 * correct_cls / total_cls if total_cls > 0 else 0
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

        print(f"Classification Accuracy: {acc:.2f}%")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print("\nConfusion Matrix:")
        print(f"TP: {true_positive} | FP: {false_positive}")
        print(f"FN: {false_negative} | TN: {true_negative}")

    if task in [Task.SEGMENTATION, Task.JOINT]:
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 1.0
        print(f"Mean IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    denoised = False
    clinical = True
    backbone = Backbone.EFFICIENTNET
    task = Task.CLASSIFICATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_only = True
    if task == task.CLASSIFICATION or task == task.JOINT:
        mask_only = False

    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.17], std=[0.21])
    ])

    dataset_path = os.path.join("..", "final_datasets", "once_more")
    if denoised:
        dataset_path = os.path.join(dataset_path, "mtl_denoised")
    else:
        dataset_path = os.path.join(dataset_path, "mtl_final")

    model = return_model(task, backbone, denoised, clinical)
    model.load_state_dict(torch.load("models/hospital/classification/efficientnet_classification_clinical.pt", weights_only=True, map_location=device))

    model.eval()
    test_dataset = MedicalImageDataset(dataset_path, split="test", mask_only=mask_only, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    test_model(model, test_loader, task, device, clinical)
