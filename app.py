import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from scripts.model import MobileNetClassifier

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model (MobileNet)
model = MobileNetClassifier(num_classes=4).to(device)
model.load_state_dict(torch.load("mobilenet_final.pth", map_location=device))

model.eval()

# Correct class order (MUST match training order)
classes = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Image transform - must match training pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI brain image to classify the tumor type using the trained MobileNet model.")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", width=350)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]

    # Show prediction
    st.success(f"### 🔍 Prediction: **{label.upper()}**")
