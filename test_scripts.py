import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

from config import Config
from dataset import MedicalImageDataset


def perform_full_test(model, val_transform):
    config = Config("config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = MedicalImageDataset(config.dataset_path, split="test", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model.to(device)
    model.eval()

    threshold = 0.3

    test_preds = []
    test_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs, labels, clinical = batch['image'].to(device), batch['label'].to(device), batch['clinical'].to(device)
            outputs = model(inputs, clinical)

            probs = torch.softmax(outputs, dim=1)
            class1_probs = probs[:, 1]

            predicted = (class1_probs > threshold).long()

            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            all_probs.extend(class1_probs.cpu().numpy())

    # Metrics
    cm = confusion_matrix(test_labels, test_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    config_name = config.to_string()

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", config_name + ".pt"))

    # Save confusion matrix plot
    os.makedirs("results/conf_matrix", exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(f"Confusion Matrix - Threshold @ {threshold}")
    plt.savefig(f"results/conf_matrix/{config_name}.png")
    plt.close()

    # Save metrics to JSON
    os.makedirs("results/metrics", exist_ok=True)
    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity
    }
    with open(f"results/metrics/{config_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save ROC curve
    os.makedirs("results/roc", exist_ok=True)
    fpr, tpr, _ = roc_curve(test_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(f"results/roc/{config_name}.png")
    plt.close()