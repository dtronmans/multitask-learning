import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from architectures.model_builder import return_model
from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClassification, \
    transfer_weights_to_clinical_model, EfficientUNetWithClinicalClassification
from dataset import MedicalImageDataset
from enums import Backbone, Task
from train_scripts.losses import DiceLossWithSigmoid


def train(train_dataloader, test_dataloader, model, task, save_path):
    print("Task: " + str(task))
    class_weights = torch.tensor([1.0, 2.0]).to(device)
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    segmentation_criterion = DiceLossWithSigmoid()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in tqdm(train_dataloader, desc="Training", leave=True):
            images, labels, masks, clinical = batch['image'].to(device), batch['label'].to(device), \
                batch['mask'].to(device), batch['clinical'].to(
                device)
            masks = (masks > 0).float()

            optimizer.zero_grad()
            predicted_seg, predicted_cls = model(images, clinical)

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
                    loss = seg_loss + 0.3 * cls_loss
                else:
                    loss = 0.3 * cls_loss
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
                images, labels, masks, clinical = batch['image'].to(device), batch['label'].to(device), \
                    batch['mask'].to(device), batch['clinical'].to(
                    device)
                masks = (masks > 0).float()

                predicted_seg, predicted_cls = model(images, clinical)

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
                        loss = seg_loss + 0.3 * cls_loss
                    else:
                        loss = 0.3 * cls_loss

                val_loss += loss.item()

                if task in [Task.CLASSIFICATION, Task.JOINT]:
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
            print(f"Train Accuracy: {train_accuracy:.2f}% - Val Accuracy: {val_accuracy:.2f}%")

    # Final model save
    print("Training complete.")
    torch.save(model.state_dict(), save_path + "_final.pt")


def construct_save_path(denoised, backbone, task):
    final_str = ""
    if backbone == Backbone.CLASSIC:
        final_str += "classic_"
    elif backbone == Backbone.EFFICIENTNET:
        final_str += "efficientnet_"
    if task == Task.JOINT:
        final_str += "joint"
    elif task == Task.CLASSIFICATION:
        final_str += "classification"
    elif task == Task.SEGMENTATION:
        final_str += "segmentation"
    if denoised is True:
        final_str += "_denoised"
    return final_str


if __name__ == "__main__":
    denoised = False
    backbone = Backbone.EFFICIENTNET
    task = Task.JOINT
    num_epochs, batch_size, learning_rate = 80, 8, 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.17], std=[0.21])
    ])

    dataset_path = os.path.join("/exports", "lkeb-hpc", "dzrogmans")
    if denoised:
        dataset_path = os.path.join(dataset_path, "mtl_denoised")
    else:
        dataset_path = os.path.join(dataset_path, "mtl_final")

    mask_only = True
    if task == task.CLASSIFICATION or task == task.JOINT:
        mask_only = False

    # model = return_model(task, backbone, denoised)


    # train joint architecture without clinical information
    pretrained_path = os.path.join("/exports", "lkeb-hpc", "dzrogmans", "models", "mmotu", "joint",
                                   "efficientnet_joint.pt")
    model = EfficientUNetWithClinicalClassification(1, 1, 8)
    model.load_state_dict(torch.load(pretrained_path))
    model.classification_head[3] = nn.Linear(128, 2)

    save_path = construct_save_path(denoised, backbone, task)
    print("Save path: " + save_path)

    train_dataset = MedicalImageDataset(dataset_path, split="train", mask_only=mask_only, transform=transform)
    val_dataset = MedicalImageDataset(dataset_path, split="val", mask_only=mask_only, transform=transform)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    print("Denoised: " + str(denoised))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train(train_loader, val_loader, model, task, save_path)
