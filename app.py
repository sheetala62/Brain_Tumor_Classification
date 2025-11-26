import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Import your ArConvNet model from scripts/model.py
from scripts.model import ArConvNet

# ---------------------------------------------------------
# DEVICE SETUP
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# LOAD ARCOVNET MODEL
# ---------------------------------------------------------
ar_model = ArConvNet(num_classes=4).to(DEVICE)
ar_model.load_state_dict(torch.load("./models/arconvnet_final.pth", map_location=DEVICE))
ar_model.eval()

# ---------------------------------------------------------
# LOAD MOBILENETV2 MODEL (trained model)
# ---------------------------------------------------------
mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")
mobilenet.classifier[1] = nn.Linear(1280, 4)
mobilenet = mobilenet.to(DEVICE)

# 🔥 Load your trained MobileNet model here
mobilenet.load_state_dict(torch.load("./models/mobilenet_final.pth", map_location=DEVICE))

mobilenet.eval()


# ---------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary']


# ---------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------
def predict(model, image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return LABELS[pred.item()]


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🧠 Brain Tumor Classification")
st.subheader("ArConvNet vs MobileNetV2 (MRI Image Upload)")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predictions
    st.write("### 🔍 Predictions:")

    ar_pred = predict(ar_model, img)
    mb_pred = predict(mobilenet, img)

    st.write(f"**ArConvNet Prediction:** {ar_pred}")
    st.write(f"**MobileNetV2 Prediction:** {mb_pred}")

    # Highlight if both agree
    if ar_pred == mb_pred:
        st.success(f"Both models agree: **{ar_pred}**")
    else:
        st.warning("Models disagree — consider reviewing the image.")

st.write("---")
st.write("Created by Pooja Hegde • Brain Tumor Classification App")
