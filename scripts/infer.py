import torch
from torchvision import transforms
from PIL import Image
from model import ArConvNet

LABELS = ['glioma','meningioma','no_tumor','pituitary']

def predict(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ArConvNet(num_classes=4).to(device)
    model.load_state_dict(torch.load('arconvnet_final.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(x).argmax(1).item()
    print(f"Prediction: {LABELS[preds]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python scripts/infer.py path/to/image.jpg")
