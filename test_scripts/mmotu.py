import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

from architectures.classification_only.efficientnet_only_classification import EfficientClassificationOnly
from architectures.classification_only.unet_classification_only import UNetClassificationOnly
from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClassification
from architectures.mtl.unet_with_classification import UNetWithClassification
from architectures.segmentation_only.efficientnet_only_segmentation import EfficientUNet
from architectures.unet_parts import BasicUNet
from dataset import MultimodalMMOTUDataset
from paired_transform import DefaultPairedTransform


def test_classification_only(model, dataloader, show=False):
    predictions = []
    actual = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            image, label, mask = batch
            image = image.to(torch.device("cpu"))
            label = label.to(torch.device("cpu"))

            # Get classification output (second output from model)
            _, class_logits = model(image)

            # Apply softmax and get predicted class
            probs = F.softmax(class_logits, dim=1)
            predicted_classes = torch.argmax(probs, dim=1)

            predictions.extend(predicted_classes.cpu().tolist())
            actual.extend(label.cpu().tolist())

            if show:
                print(f"Predicted: {predicted_classes.item()}, Actual: {label.item()}")

    # Calculate accuracy
    correct = sum(p == a for p, a in zip(predictions, actual))
    total = len(actual)
    accuracy = correct / total if total > 0 else 0
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")


def test_segmentation_only(model, dataloader, show=False):
    ious = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image, label, mask = batch
            predicted, _ = model(image)
            predicted_mask = (torch.sigmoid(predicted[0]) > 0.5).float()
            gt_mask = (mask > 0).float()
            intersection = (predicted_mask * gt_mask).sum()
            union = (predicted_mask + gt_mask - predicted_mask * gt_mask).sum()

            iou = (intersection + eps) / (union + eps)
            # print("IOU: " + str(iou.item()))
            ious.append(iou.item())

        print("Average IOU: " + str(round(sum(ious) / len(ious), 2)))


if __name__ == "__main__":
    directory = "../final_datasets/once_more/OTU_2d"
    eps = 1e-6
    dataset = MultimodalMMOTUDataset(directory, phase="test", paired_transform=DefaultPairedTransform())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNetWithClassification(1, 1,  8)
    model.load_state_dict(
        torch.load("models/mmotu/joint/unet_joint.pt", map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))

    model.eval()

    test_segmentation_only(model, dataloader)
