import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ArConvNet

LABEL_MAP = {'glioma':0, 'meningioma':1, 'no_tumor':2, 'pituitary':3}

class BrainTumorDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['image_path']
        label = LABEL_MAP[self.df.iloc[idx]['label']]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    df = pd.read_csv('dataset_dataframe.csv')
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = BrainTumorDataset(train_df, transform)
    test_ds = BrainTumorDataset(test_df, transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ArConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses, accs = [], []
    for epoch in range(5):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f} Acc={epoch_acc:.4f}")
        losses.append(epoch_loss)
        accs.append(epoch_acc)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.plot(losses); plt.title("Loss")
    plt.subplot(1,2,2); plt.plot(accs); plt.title("Accuracy")
    plt.savefig("training_curves.png")
    torch.save(model.state_dict(), "arconvnet_final.pth")
    print("Training complete, model saved as arconvnet_final.pth")

if __name__ == "__main__":
    main()
