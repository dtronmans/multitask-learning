import torch
from sklearn.metrics import accuracy_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from architectures.mtl import MTLNet
from config import Config
from dataset import MedicalImageDataset
from test_scripts import perform_full_test

if __name__ == "__main__":
    config = Config("config.json")
    num_epochs, batch_size, learning_rate = config.num_epochs, config.batch_size, config.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MTLNet(1, 1, 2)
    model.to(device)
    if config.cropped:
        resize = transforms.Resize((164, 164))
    else:
        resize = transforms.Resize((336, 544))

    transform = transforms.Compose([
        resize,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((164, 164)),
        transforms.ToTensor(),
    ])

    mask_only = config.task == "segmentation"

    train_dataset = MedicalImageDataset("config.dataset_path", split="train", mask_only=mask_only, transform=transform)
    val_dataset = MedicalImageDataset("config.dataset_path", split="val", mask_only=mask_only, transform=transform)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_weights = torch.tensor([1.0, 2.0]).to(device)
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    segmentation_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_cls_loss, train_seg_loss = 0.0, 0.0
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader):
            inputs, labels, masks, clinical = batch['image'].to(device), batch['label'].to(device), batch['mask'].to(device), batch['clinical'].to(device)
            optimizer.zero_grad()
            seg_logits, class_logits = model(inputs)
            cls_loss = classification_criterion(class_logits, labels)

            valid_mask_indices = torch.any(masks != 0, dim=(2, 3)).squeeze(1)
            if valid_mask_indices.any() and config.task != "classification":
                valid_seg_logits = seg_logits[valid_mask_indices]
                valid_masks = masks[valid_mask_indices].float()
                seg_loss = segmentation_criterion(valid_seg_logits, valid_masks)
            else:
                seg_loss = torch.tensor(0.0, device=device)

            total_loss = cls_loss + seg_loss
            total_loss.backward()
            optimizer.step()

            train_cls_loss += cls_loss.item()
            train_seg_loss += seg_loss.item()

            _, predicted = torch.max(class_logits, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_recall = recall_score(train_labels, train_preds, average='macro')

        model.eval()
        val_cls_loss, val_seg_loss = 0.0, 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs, labels, masks = batch['image'].to(device), batch['label'].to(device), batch['mask'].to(device)
                seg_logits, class_logits = model(inputs)
                cls_loss = classification_criterion(class_logits, labels)

                valid_mask_indices = torch.any(masks != 0, dim=(2, 3))
                if valid_mask_indices.any() and config.task != "classification":
                    valid_seg_logits = seg_logits[valid_mask_indices]
                    valid_masks = masks[valid_mask_indices].float()
                    seg_loss = segmentation_criterion(valid_seg_logits, valid_masks)
                else:
                    seg_loss = torch.tensor(0.0, device=device)

                val_cls_loss += cls_loss.item()
                val_seg_loss += seg_loss.item()

                _, predicted = torch.max(class_logits, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

            val_acc = accuracy_score(val_labels, val_preds)
            val_recall = recall_score(val_labels, val_preds, average='macro')

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"| Train CLS Loss: {train_cls_loss / len(train_loader):.4f} "
              f"| Train SEG Loss: {train_seg_loss / len(train_loader):.4f} "
              f"| Train Acc: {train_acc:.4f} "
              f"| Train Recall: {train_recall:.4f} "
              f"|| Val CLS Loss: {val_cls_loss / len(val_loader):.4f} "
              f"| Val SEG Loss: {val_seg_loss / len(val_loader):.4f} "
              f"| Val Acc: {val_acc:.4f} "
              f"| Val Recall: {val_recall:.4f}")

    perform_full_test(model, val_transform)
