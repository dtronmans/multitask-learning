import torch
from torch import nn


class EfficientNetWithClinical(nn.Module):
    def __init__(self, base_model, clinical_feature_dim=2, num_classes=2):
        super(EfficientNetWithClinical, self).__init__()

        # 🔥 Extract image feature dim BEFORE replacing the classifier
        image_feat_dim = base_model.classifier[1].in_features

        # 🔁 Remove original classifier
        base_model.classifier = nn.Identity()
        self.base_model = base_model

        # 🧠 New classifier using image features + clinical features
        self.classifier = nn.Sequential(
            nn.Linear(image_feat_dim + clinical_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, clinical_features):
        image_features = self.base_model(x)
        x = torch.cat((image_features, clinical_features), dim=1)
        out = self.classifier(x)
        return out
