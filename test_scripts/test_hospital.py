import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, precision_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from architectures.model_with_temperature import ModelWithTemperature
from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClinicalClassification
from dataset import MedicalImageDataset
from enums import Backbone, Task
from train_scripts.mtl_training import return_model




def test_model(model, dataloader, task, device, clinical, threshold=0.5, show=False):
    model.to(device)
    model.eval()
    correct_cls = 0
    total_cls = 0
    iou_scores = []

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=True):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)
            clinical_info = batch['clinical'].to(device)
            image_paths = batch['image_path']
            masks = (masks > 0).float()

            if clinical:
                predicted_seg, predicted_cls = model(images, clinical_info)
            else:
                if task == Task.CLASSIFICATION:
                    predicted_cls = model(images)
                else:
                    predicted_seg, predicted_cls = model(images)

            if task in [Task.CLASSIFICATION, Task.JOINT]:
                probs = torch.softmax(predicted_cls, dim=1)[:, 1]
                preds = (probs >= threshold).long()

                correct_cls += (preds == labels).sum().item()
                total_cls += labels.size(0)

                true_positive += ((preds == 1) & (labels == 1)).sum().item()
                true_negative += ((preds == 0) & (labels == 0)).sum().item()
                false_positive += ((preds == 1) & (labels == 0)).sum().item()
                false_negative += ((preds == 0) & (labels == 1)).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            if task in [Task.SEGMENTATION, Task.JOINT]:
                pred_mask = torch.sigmoid(predicted_seg) > 0.5
                pred_mask = pred_mask.float()
                intersection = (pred_mask * masks).sum()
                union = (pred_mask + masks).clamp(0, 1).sum()
                iou = intersection / union if union != 0 else torch.tensor(1.0)
                iou_scores.append(iou.item())

            if show and task == Task.JOINT:
                visualize_joint_prediction(images, pred_mask, preds, labels, clinical_info, image_paths)

    if task in [Task.CLASSIFICATION, Task.JOINT]:
        acc = 100 * correct_cls / total_cls if total_cls > 0 else 0
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')  # avoid AUC error if only one class
        precision = precision_score(all_labels, all_preds)

        print(f"Classification Accuracy: {acc:.2f}%")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\nConfusion Matrix:")
        print(f"TP: {true_positive} | FP: {false_positive}")
        print(f"FN: {false_negative} | TN: {true_negative}")

    if task in [Task.SEGMENTATION, Task.JOINT]:
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 1.0
        print(f"Mean IoU: {mean_iou:.4f}")


def visualize_joint_prediction(images, pred_masks, preds, labels, clinical_info, image_paths):
    images = images.cpu()
    pred_masks = pred_masks.cpu()
    preds = preds.cpu()
    labels = labels.cpu()
    clinical_info = clinical_info.cpu()

    def label_text(val):
        return "Benign" if val == 0 else "Malignant"

    batch_size = min(1, images.size(0))  # Show only the first image
    for i in range(batch_size):
        image_tensor = images[i]
        if image_tensor.shape[0] == 1:
            image = image_tensor[0].numpy()  # Grayscale
            cmap = 'gray'
        else:
            image = image_tensor.permute(1, 2, 0).numpy()  # RGB
            cmap = None

        pred_mask = pred_masks[i][0].numpy()
        pred_label = label_text(preds[i].item())
        true_label = label_text(labels[i].item())
        clinical = clinical_info[i].int().tolist()

        menopausal_status = "pre" if clinical[0] == 0 else "post"
        oncology_center = "true" if clinical[1] == 0 else "false"
        clinical_text = f"menopausal status: {menopausal_status} | oncology center: {oncology_center}"
        image_path = image_paths[i] if isinstance(image_paths[i], str) else image_paths[i].decode('utf-8')

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(
            f"Prediction: {pred_label} | True: {true_label} | \n{clinical_text}",
            fontsize=14,
        )

        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(image, cmap=cmap)
        axs[1].imshow(pred_mask, alpha=0.5, cmap='Reds')
        axs[1].set_title("Predicted Mask Overlay")
        axs[1].axis("off")

        plt.figtext(0.5, 0.01, f"Path: {image_path}", wrap=True, ha='center', fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    denoised = False
    clinical = False
    backbone = Backbone.EFFICIENTNET
    task = Task.SEGMENTATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_only = True
    if task == task.CLASSIFICATION or task == task.JOINT:
        mask_only = False

    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor()
    ])

    dataset_path = os.path.join("..", "final_datasets", "once_more")
    if denoised:
        dataset_path = os.path.join(dataset_path, "mtl_denoised")
    else:
        dataset_path = os.path.join(dataset_path, "mtl_final")

    print("dataset path: " + dataset_path)
    model = return_model(task, backbone, denoised, clinical)
    model.load_state_dict(torch.load("models/hospital/segmentation/efficientnet_segmentation.pt", weights_only=True,
                                     map_location=device))

    model.eval()
    test_dataset = MedicalImageDataset(dataset_path, split="test", mask_only=mask_only, transform=transform)
    validation_dataset = MedicalImageDataset(dataset_path, split="test", mask_only=mask_only, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    test_model(model, test_loader, task, device, clinical, show=False)
