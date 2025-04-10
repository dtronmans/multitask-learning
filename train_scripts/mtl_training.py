from collections import Counter

import torch
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

from architectures.mtl import MTLNet
from dataset import MedicalImageDataset

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 4
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MTLNet(1, 1, 2)
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MedicalImageDataset("/exports/lkeb-hpc/dzrogmans/lumc_rdg_final", transform=transform)

    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=[dataset[idx]['label'] for idx in range(len(dataset))],
        random_state=42
    )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_targets = [dataset[idx]['label'] for idx in train_indices]
    label_counts = Counter(train_targets)

    total = sum(label_counts.values())
    class_weights = [total / label_counts[i] for i in range(len(label_counts))]
    weights_tensor = torch.FloatTensor(class_weights).to(device)

    classification_criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    segmentation_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_cls_loss, train_seg_loss = 0, 0
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader):
            inputs, labels, masks = batch['image'].to(device), batch['label'].to(device), batch['mask'].to(device)
            optimizer.zero_grad()
            seg_logits, class_logits = model(inputs)
            cls_loss = classification_criterion(class_logits, labels)

            valid_mask_indices = torch.any(masks != 0, dim=(1, 2))
            if valid_mask_indices.any():
                valid_seg_logits = seg_logits[valid_mask_indices]
                valid_masks = masks[valid_mask_indices].unsqueeze(1).float()  # Add channel dim
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
        val_cls_loss, val_seg_loss = 0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs, labels, masks = batch['image'].to(device), batch['label'].to(device), batch['mask'].to(device)
                seg_logits, class_logits = model(inputs)
                cls_loss = classification_criterion(class_logits, labels)

                valid_mask_indices = torch.any(masks != 0, dim=(1, 2))
                if valid_mask_indices.any():
                    valid_seg_logits = seg_logits[valid_mask_indices]
                    valid_masks = masks[valid_mask_indices].unsqueeze(1).float()
                    seg_loss = segmentation_criterion(valid_seg_logits, valid_masks)
                else:
                    seg_loss = torch.tensor(0.0, device=device)

                val_cls_loss += cls_loss.item()
                val_seg_loss += seg_loss.item()

                _, predicted = torch.max(class_logits, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Train CLS Loss: {train_cls_loss/len(train_loader):.4f} "
              f"| Train SEG Loss: {train_seg_loss/len(train_loader):.4f} "
              f"| Train Acc: {train_acc:.4f} "
              f"| Train Recall: {train_recall:.4f} "
              f"|| Val CLS Loss: {val_cls_loss/len(val_loader):.4f} "
              f"| Val SEG Loss: {val_seg_loss/len(val_loader):.4f} "
              f"| Val Acc: {val_acc:.4f} "
              f"| Val Recall: {val_recall:.4f}")
