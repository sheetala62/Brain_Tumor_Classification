import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNetV2, mobilenet_v2

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetClassifier, self).__init__()

        # Load base MobileNetV2
        self.mobilenet = mobilenet_v2(weights=None)

        # Replace last classifier layer
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)
