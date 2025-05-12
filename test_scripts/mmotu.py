import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from architectures.classification_only.unet_classification_only import UNetClassificationOnly
from architectures.mtl.efficientnet_with_classification import EfficientUNetWithClassification
from architectures.mtl.unet_with_classification import UNetWithClassification
from architectures.segmentation_only.efficientnet_only_segmentation import EfficientUNet
from architectures.unet_parts import BasicUNet
from dataset import MultimodalMMOTUDataset


def test_classification_only(model, dataloader, show=False):
    predictions = []
    actual = []

    with torch.no_grad():
        for batch in dataloader:
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
        for batch in dataloader:
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
    directory = "OTU_2d_denoised"
    eps = 1e-6
    transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((336, 544)),
        transforms.ToTensor()
    ])

    dataset = MultimodalMMOTUDataset(directory, phase="test", transforms=transform, mask_transforms=mask_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = BasicUNet(1, 1)
    # model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # original_conv = model.features[0][0]
    # model.features[0][0] = nn.Conv2d(
    #     in_channels=1,
    #     out_channels=original_conv.out_channels,
    #     kernel_size=original_conv.kernel_size,
    #     stride=original_conv.stride,
    #     padding=original_conv.padding,
    #     bias=original_conv.bias is not None
    # )
    model.load_state_dict(
        torch.load("models/mmotu/segmentation/classic_unet_denoised.pt", map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))

    model.eval()

    test_segmentation_only(model, dataloader)

    # now we need segmentation only, classification only and joint testing
