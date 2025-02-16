"""
The "Design" part of the website was generated with the help of the AI v0 by Vercel.

A Streamlit website for simple User Interface.
You can start the Streamlit-Website by pasting following command in your cmd:
python -m streamlit run [path to this script]
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import torch
from cnn import *
from predictions import predict

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)

# load the trained Model and make it ready for predictions
model = ColorizationCNN()
model_ = os.path.join(os.getcwd(), "colorization_model_with_rebalancing.pth")
model.eval()
state_dict = torch.load(model_, map_location=torch.device("cpu"))

new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(new_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()


def convert_to_bw(image):
    image = np.array(image)
    if len(image.shape) == 2:
        bw_image = image
    else:
        bw_image = cv2.cvtColor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return bw_image


def colorize_image(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    out_np = np.asarray(Image.open(temp_path))
    if out_np.ndim == 2:
        out_np = np.tile(out_np[:, :, None], 3)

    original_shape = out_np.shape

    predicted_image = predict(model, temp_path, device=device)
    predicted_image = (predicted_image * 255).astype("uint8")

    resized_predicted_img = cv2.resize(predicted_image, (original_shape[1],original_shape[0]), interpolation=cv2.INTER_AREA)

    return resized_predicted_img


st.set_page_config(page_title="Image Colorization", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: #001845;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #4ea8de;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: #ffffff;
        border: none;
        border-radius: 30px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .upload-section, .result-section {
        background: rgba(52, 152, 219, 0.1);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stRadio > label {
        color: #ffffff;
    }
    .stFileUploader > label {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Image Colorization</h1>", unsafe_allow_html=True)

upload_option = st.radio("Wähle Input-Methode", ("Bild Hochladen", "Foto machen"), horizontal=True)

if upload_option == "Bild Hochladen":
    uploaded_file = st.file_uploader("Wähle ein Bild", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    uploaded_file = st.camera_input("Mache ein Foto")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if 'image' in locals() and image is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h3 style='text-align: center; color: #ffffff;'>Groundtruth Bild</h3>", unsafe_allow_html=True)
        st.image(image)

    with col2:
        st.markdown("<h3 style='text-align: center; color: #ffffff;'>Schwarz & Weiß</h3>", unsafe_allow_html=True)
        bw_image = convert_to_bw(image)
        st.image(bw_image)

    with col3:
        st.markdown("<h3 style='text-align: center; color: #9d4edd;'>Eingefärbtes Bild</h3>", unsafe_allow_html=True)
        if st.button("Einfärben"):
            colorized_image = colorize_image(image)
            st.image(colorized_image)
        else:
            st.write("Klicke auf 'Einfärben' um ein Ergebniss zu erhalten")

else:
    st.write("Upload an image or take a picture to start the colorization process.")
    