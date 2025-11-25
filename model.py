import torch.nn as nn
import torchvision.models as models

class TransferModel(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)  # binary output
        elif backbone == "resnet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)
        else:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)
