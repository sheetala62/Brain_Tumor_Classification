import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# Dataset Loader
# -----------------------
LABEL_MAP = {'glioma':0, 'meningioma':1, 'no_tumor':2, 'pituitary':3}

class BrainTumorDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform

        for label in os.listdir(root):
            folder = os.path.join(root, label)
            if os.path.isdir(folder):
                for img in os.listdir(folder):
                    if img.lower().endswith(('.jpg','.png','.jpeg')):
                        self.paths.append(os.path.join(folder, img))
                        self.labels.append(LABEL_MAP[label])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------
# ArConvNet (your model)
# -----------------------
class ArConvNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.gap  = nn.AdaptiveAvgPool2d((7,7))

        self.fc1 = nn.Linear(128*7*7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------
# Evaluation Function
# -----------------------
def evaluate(model, loader, device):
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, preds = torch.max(out,1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print("\nAccuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_true,y_pred,
          target_names=['glioma','meningioma','no_tumor','pituitary']))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.show()

    return acc

# -----------------------
# Main function
# -----------------------
def main():

    TEST_DIR = "./brisc2025/classification_task/test"
    ARCONV_MODEL_PATH = "./models/arconvnet_final.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    test_ds = BrainTumorDataset(TEST_DIR, transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Load ArConvNet
    ar_model = ArConvNet().to(device)
    ar_model.load_state_dict(torch.load(ARCONV_MODEL_PATH, map_location=device))
    ar_model.eval()

    # Load MobileNetV2
    mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
    mobilenet.classifier[1] = nn.Linear(1280, 4)
    mobilenet = mobilenet.to(device)
    mobilenet.eval()

    print("\n===== Evaluating ArConvNet =====")
    acc_ar = evaluate(ar_model, test_loader, device)

    print("\n===== Evaluating MobileNetV2 =====")
    acc_mb = evaluate(mobilenet, test_loader, device)

    print("\n===== FINAL COMPARISON =====")
    print(f"ArConvNet Accuracy:   {acc_ar:.4f}")
    print(f"MobileNetV2 Accuracy: {acc_mb:.4f}")

if __name__ == "__main__":
    main()

