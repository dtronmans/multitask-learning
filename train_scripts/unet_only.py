import torch.cuda
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from architectures.unet_parts import BasicUNet
from dataset import MedicalImageDataset

if __name__ == "__main__":
    num_epochs = 100
    batch_size = 4
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicUNet(1, 1)
    model.load_state_dict(torch.load("models/mmotu/mmotu_unet_grayscale.pt", map_location=device))

    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MedicalImageDataset("../final_datasets/lumc_rdg_final", transform=transform, mask_only=True)

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

    segmentation_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            inputs, masks = batch['image'].to(device), batch['mask'].to(device)
            masks = (masks > 0.5).float()
            optimizer.zero_grad()
            seg_logits = model(inputs)
            seg_loss = segmentation_criterion(seg_logits, masks)
            seg_loss.backward()
            optimizer.step()
            train_loss += seg_loss.item() * inputs.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                inputs, masks = batch['image'].to(device), batch['mask'].to(device)
                masks = (masks > 0.5).float()
                seg_logits = model(inputs)
                seg_loss = segmentation_criterion(seg_logits, masks)
                val_loss += seg_loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}")
