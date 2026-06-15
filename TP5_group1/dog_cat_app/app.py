import streamlit as st
import torch
import numpy as np
import cv2 as cv
import os

# Set page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

def my_ANN_eval(X, weights):
    """
    ANN inference using PyTorch tensors.
    weights order: [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5]
    """
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = weights

    Z1 = torch.matmul(X, W1) + b1
    A1 = torch.relu(Z1)

    Z2 = torch.matmul(A1, W2) + b2
    A2 = torch.relu(Z2)

    Z3 = torch.matmul(A2, W3) + b3
    A3 = torch.relu(Z3)

    Z4 = torch.matmul(A3, W4) + b4
    A4 = torch.relu(Z4)

    Z5 = torch.matmul(A4, W5) + b5
    return Z5

@st.cache_resource
def load_model_data():
    model_path = 'model_data.pt'
    if not os.path.exists(model_path):
        return None

    # Load everything from the single .pt file
    data = torch.load(model_path, map_location=torch.device('cpu'))
    return data

# UI
st.title("🐾 Cat vs Dog Classifier")
st.markdown("""
This app classifies images of cats and dogs using an **ANN** and **PCA**
trained in `ex5.ipynb`.
""")

# Load models
model_data = load_model_data()

if model_data is None:
    st.error("⚠️ **'model_data.pt' not found!**")
    st.info("""
    To generate the required file, run this code in a new cell at the end of **ex5.ipynb**:

    ```python
    import torch

    # 1. Prepare PCA data (V matrix and mean)
    pca_data = {
        'components': torch.tensor(pca.components_, dtype=torch.float32),
        'mean': torch.tensor(pca.mean_, dtype=torch.float32)
    }

    # 2. Prepare ANN weights
    if 'best_weights' in locals() and best_weights is not None:
        weights = [w.detach().cpu() for w in best_weights]
    else:
        weights = [p.detach().cpu() for p in params]

    # 3. Save everything to one file
    torch.save({
        'pca': pca_data,
        'weights': weights
    }, 'model_data.pt')
    print("model_data.pt saved successfully!")
    ```

    Then, move `model_data.pt` into the `dog_cat_app` folder.
    """)
else:
    pca_comp = model_data['pca']['components']
    pca_mean = model_data['pca']['mean']
    weights = model_data['weights']

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner('Classifying...'):
            # 1. Image Preprocessing (matches Step 2 & 3 of notebook)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            resized = cv.resize(gray, (128, 128))
            flattened = torch.tensor(resized.reshape(1, -1), dtype=torch.float32)
            normalized = flattened / 255.0

            # 2. PCA Transformation manually using the V matrix (components)
            # x_pca = (x - mean) @ components.T
            x_centered = normalized - pca_mean
            x_pca = torch.matmul(x_centered, pca_comp.t())

            # 3. ANN Inference
            with torch.no_grad():
                logits = my_ANN_eval(x_pca, weights)
                probs = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1).item()
                confidence = probs[0][prediction].item()

            with col2:
                label = "Dog" if prediction == 1 else "Cat"
                st.metric("Prediction", label)
                st.metric("Confidence", f"{confidence:.2%}")

                if prediction == 1:
                    st.write("🐶 Woof! It's a dog.")
                else:
                    st.write("🐱 Meow! It's a cat.")

st.divider()

# Creators Section
st.subheader("👥 Project Creators")
cols = st.columns(4)

creators = [
    {"name": "NUON CHANVATHANA", "img": "e20230118.jpg"},
    {"name": "LY LAISRUN", "img": "image.png"},
    {"name": "Roeurn Phannet", "img": "ROEURN PHANNET.jpg"},
    {"name": "LY BUNLENG", "img": "e20230130.jpg"}
]

for col, creator in zip(cols, creators):
    with col:
        if os.path.exists(creator["img"]):
            st.image(creator["img"], use_container_width=True)
        else:
            st.warning(f"Image not found: {creator['img']}")
        st.caption(f"**{creator['name']}**")

st.divider()
st.caption("Developed with Streamlit & PyTorch")
