from collections import Counter

import torch
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

from dataset import MedicalImageDataset

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 4
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 8)
    )
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load("models/mmotu/resnet18.pt", map_location=device))
    model.fc = nn.Sequential(
        nn.Linear(512, 2)
    )
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.RandomApply([transforms.RandomHorizontalFlip(p=1.0)], p=0.3),
        transforms.RandomApply([transforms.RandomRotation(degrees=20)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = MedicalImageDataset(root_dir='lkeb-hpc/exports/dzrogmans/lumc_rdg_final', split='train', transform=transform)
    val_dataset = MedicalImageDataset(root_dir='lkeb-hpc/exports/dzrogmans/lumc_rdg_final', split='val', transform=val_transform)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        for batch in tqdm(train_loader):
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
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
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds, average='macro')

        print(f"Epoch [{epoch+1}/{num_epochs}]"
              f" Train Loss: {train_loss/len(train_loader):.4f}"
              f" Acc: {train_acc:.4f}"
              f" Recall: {train_recall:.4f}"
              f" | Val Loss: {val_loss/len(val_loader):.4f}"
              f" Acc: {val_acc:.4f}"
              f" Recall: {val_recall:.4f}")