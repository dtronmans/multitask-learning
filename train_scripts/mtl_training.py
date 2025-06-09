import argparse
import os

import torch
from sklearn.metrics import recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from architectures.model_builder import return_model
from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClassification, \
    transfer_weights_to_clinical_model
from dataset import MedicalImageDataset
from enums import Backbone, Task
from paired_transform import PairedTransform
from train_scripts.losses import DiceLossWithSigmoid


def train(train_dataloader, test_dataloader, model, task, save_path, clinical):
    print("Task: " + str(task))
    labels = []
    for batch in train_dataloader:
        labels.append(batch['label'])
    class_weights = torch.tensor([1.0, 2.0]).to(device)
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    segmentation_criterion = DiceLossWithSigmoid()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_loss = np.inf
    for epoch in range(num_epochs):
        all_labels = []
        all_preds = []
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in tqdm(train_dataloader, desc="Training", leave=True):
            images, labels, masks, clinical_info = batch['image'].to(device), batch['label'].to(device), \
                batch['mask'].to(device), batch['clinical'].to(
                device)
            masks = (masks > 0).float()

            optimizer.zero_grad()
            if clinical:
                predicted_seg, predicted_cls = model(images, clinical_info)
            else:
                predicted_seg, predicted_cls = model(images)

            if task == Task.CLASSIFICATION:
                loss = classification_criterion(predicted_cls, labels)
            elif task == Task.SEGMENTATION:
                loss = segmentation_criterion(predicted_seg, masks)
            elif task == Task.JOINT:
                cls_loss = classification_criterion(predicted_cls, labels)
                valid_mask_indices = (masks.flatten(1).sum(dim=1) > 0)  # shape: (batch_size,)

                if valid_mask_indices.any():
                    valid_predicted_seg = predicted_seg[valid_mask_indices]
                    valid_masks = masks[valid_mask_indices]
                    seg_loss = segmentation_criterion(valid_predicted_seg, valid_masks)
                    loss = 0.3 * seg_loss + cls_loss
                else:
                    loss = cls_loss
            else:
                raise ValueError(f"Unsupported task type: {task}")

            train_loss += loss.item()

            if task in [Task.CLASSIFICATION, Task.JOINT]:
                preds = torch.argmax(predicted_cls, dim=1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Validation", leave=True):
                images, labels, masks, clinical_info = batch['image'].to(device), batch['label'].to(device), \
                    batch['mask'].to(device), batch['clinical'].to(
                    device)
                masks = (masks > 0).float()

                if clinical:
                    predicted_seg, predicted_cls = model(images, clinical_info)
                else:
                    predicted_seg, predicted_cls = model(images)

                if task == Task.CLASSIFICATION:
                    loss = classification_criterion(predicted_cls, labels)
                elif task == Task.SEGMENTATION:
                    loss = segmentation_criterion(predicted_seg, masks)
                elif task == Task.JOINT:
                    cls_loss = classification_criterion(predicted_cls, labels)
                    valid_mask_indices = (masks.flatten(1).sum(dim=1) > 0)  # shape: (batch_size,)

                    if valid_mask_indices.any():
                        valid_predicted_seg = predicted_seg[valid_mask_indices]
                        valid_masks = masks[valid_mask_indices]
                        seg_loss = segmentation_criterion(valid_predicted_seg, valid_masks)
                        loss = 0.3 * seg_loss + cls_loss
                    else:
                        loss = cls_loss

                val_loss += loss.item()

                if task in [Task.CLASSIFICATION, Task.JOINT]:
                    probs = torch.softmax(predicted_cls, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_labels.append(labels.cpu())
                    all_preds.append(preds.cpu())

                    preds = torch.argmax(predicted_cls, dim=1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

        avg_val_loss = val_loss / len(test_dataloader)
        val_accuracy = 100 * correct_val / total_val if total_val > 0 else 0

        if avg_val_loss < best_val_loss:
            print("Saving best model so far!")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path + ".pt")
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if task in [Task.CLASSIFICATION, Task.JOINT]:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
            print(f"Validation Recall (Sensitivity): {recall:.4f}")
            print(f"Train Accuracy: {train_accuracy:.2f}% - Val Accuracy: {val_accuracy:.2f}%")

    print("Training complete.")


def construct_save_path(denoised, backbone, task, clinical):
    final_str = ""
    if backbone == Backbone.CLASSIC:
        final_str += "classic_"
    elif backbone == Backbone.EFFICIENTNET:
        final_str += "efficientnet_"
    elif backbone == Backbone.RESNET:
        final_str += "resnet_"
    if task == Task.JOINT:
        final_str += "joint"
    elif task == Task.CLASSIFICATION:
        final_str += "classification"
    elif task == Task.SEGMENTATION:
        final_str += "segmentation"
    if clinical is True:
        final_str += "_clinical"
    if denoised is True:
        final_str += "_denoised"
    return final_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model configuration")

    parser.add_argument('--clinical', dest='clinical', action='store_true',
                        help='Include clinical data (default: True)')
    parser.add_argument('--no-clinical', dest='clinical', action='store_false', help='Exclude clinical data')

    parser.add_argument('--task', type=str, choices=[t.value for t in Task], default='JOINT', help='Task type')
    parser.add_argument('--backbone', type=str, choices=[b.value for b in Backbone], default='EFFICIENTNET',
                        help='Backbone type')

    parser.set_defaults(clinical=True)

    args = parser.parse_args()

    denoised = False
    cropped = False
    clinical = args.clinical
    backbone = Backbone(args.backbone)
    task = Task(args.task)
    num_epochs, batch_size, learning_rate = 80, 8, 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join("/exports", "lkeb-hpc", "dzrogmans")
    if denoised:
        dataset_path = os.path.join(dataset_path, "mtl_denoised")
    elif cropped:
        dataset_path = os.path.join(dataset_path, "mtl_cropped")
    else:
        dataset_path = os.path.join(dataset_path, "mtl_final")

    mask_only = True
    if task == task.CLASSIFICATION or task == task.JOINT:
        mask_only = False

    model = return_model(task, backbone, denoised, clinical)

    save_path = construct_save_path(denoised, backbone, task, clinical)
    print("Save path: " + save_path)
    pair_transform = PairedTransform(size=(336, 544))
    if cropped:
        pair_transform = PairedTransform(size=(164, 164))

    train_dataset = MedicalImageDataset(dataset_path, split="train", mask_only=mask_only, transform=pair_transform)
    val_dataset = MedicalImageDataset(dataset_path, split="val", mask_only=mask_only, cropped=cropped)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    print("Denoised: " + str(denoised))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train(train_loader, val_loader, model, task, save_path, clinical)
