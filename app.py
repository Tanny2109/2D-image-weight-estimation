import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from model import MultiViewNutritionModel

# Load mean/std for de-normalization
means = pd.read_csv("means.csv", index_col=0).squeeze()
stds = pd.read_csv("stds.csv", index_col=0).squeeze()

# Load model
device = torch.device('cpu')
model = MultiViewNutritionModel(output_dim=12, freeze_backbone=True)
model.load_state_dict(torch.load('chkpt_2.pth', map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

target_names = list(means.index)

st.title("🍎 Food Nutrition Estimator")
st.write("Upload an image of food to estimate weight and nutritional values.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner('Estimating...'):
        img = transform(image).unsqueeze(0)               # [1, 3, 128, 128]
        img = img.repeat(37, 1, 1, 1)                      # [37, 3, 128, 128]
        img = img.unsqueeze(0)                             # [1, 37, 3, 128, 128]
        with torch.no_grad():
            pred = model(img).squeeze().numpy()
            denorm_pred = (pred * stds.values) + means.values
        result = dict(zip(target_names, denorm_pred.round(2)))
    
    st.success("Estimated Nutrition Values:")
    st.table(result)

