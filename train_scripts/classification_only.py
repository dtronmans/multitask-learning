from collections import Counter

import torch
from sklearn.metrics import accuracy_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import copy

from dataset import MedicalImageDataset

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 4
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b1()

    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    model.load_state_dict(torch.load("mmotu_efficientnet_b1.pt", map_location=device))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=1.0),
        ], p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=20)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = MedicalImageDataset("/exports/lkeb-hpc/dzrogmans/mtl_denoised", split="train", transform=transform)
    val_dataset = MedicalImageDataset("/exports/lkeb-hpc/dzrogmans/mtl_denoised", split="val", transform=transform)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # train_targets = [sample['label'] for sample in train_dataset]
    #
    # label_counts = Counter(train_targets)
    # print(label_counts)
    #
    # total = sum(label_counts.values())
    # num_classes = max(label_counts.keys())
    #
    # class_weights = [total / label_counts[i] if i in label_counts else 0.0 for i in range(num_classes)]
    # weights_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        for batch in tqdm(train_loader):
            inputs, labels, clinical = batch['image'].to(device), batch['label'].to(device), batch['clinical'].to(
                device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_recall = recall_score(train_labels, train_preds, average='macro')

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs, labels, clinical = batch['image'].to(device), batch['label'].to(device), batch['clinical'].to(
                    device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds, average='macro')

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}]"
              f" Train Loss: {train_loss / len(train_loader):.4f}"
              f" Acc: {train_acc:.4f}"
              f" Recall: {train_recall:.4f}"
              f" | Val Loss: {avg_val_loss:.4f}"
              f" Acc: {val_acc:.4f}"
              f" Recall: {val_recall:.4f}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            print("Saved the best model!")
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
