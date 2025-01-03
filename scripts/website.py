import streamlit as st
import numpy as np
from PIL import Image
import cv2


def convert_to_bw(image):
    return cv2.cvtColor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)


def colorize_image(image):
    # In this function the image will get colorized with the AI
    return image


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
    image = st.camera_input("Mache ein Foto")
    if image is not None:
        image = Image.open(image)

if 'image' in locals() and image is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h3 style='text-align: center; color: #ffffff;'>Groundtruth Bild</h3>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("<h3 style='text-align: center; color: #ffffff;'>Schwarz & Weiß</h3>", unsafe_allow_html=True)
        bw_image = convert_to_bw(image)
        st.image(bw_image, use_column_width=True)

    with col3:
        st.markdown("<h3 style='text-align: center; color: #9d4edd;'>Eingefärbtes Bild</h3>", unsafe_allow_html=True)
        if st.button("Einfärben"):
            colorized_image = colorize_image(bw_image)
            st.image(colorized_image, use_column_width=True)
        else:
            st.write("Klicke auf 'Einfärben' um ein Ergebniss zu erhalten")

else:
    st.write("Upload an image or take a picture to start the colorization process.")


