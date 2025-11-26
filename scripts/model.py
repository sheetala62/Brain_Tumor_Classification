import torch
import torch.nn as nn
import torch.nn.functional as F

class ArConvNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ArConvNet, self).__init__()

        # Convolution Layers (same as your notebook)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling Layers
        self.pool = nn.MaxPool2d(2, 2)

        # FIXED output shape 128×7×7
        self.gap = nn.AdaptiveAvgPool2d((7, 7))

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))

        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))

        # Adaptive pooling to 7×7
        x = self.gap(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x