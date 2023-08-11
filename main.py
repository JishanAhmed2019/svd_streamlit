import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="SVD - Singular Value Decomposition",
    page_icon="./static/icons8-matrix-16.png",
)

import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sidebar_utils import common_markdown
from streamlit_lottie import st_lottie_spinner
import json

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def do_svd(x):
    U, S, VT = np.linalg.svd(x, full_matrices=False)
    return U, S, VT


@st.cache_data(show_spinner=False)
def reconstruct(U, S, VT, rank):
    Xprox = U[:, :rank] @ S[:rank, :rank] @ VT[:rank, :]
    return Xprox

@st.cache_data(show_spinner=False)
def load_json(path):
    with open(path, 'rb') as f:
        json_str = json.load(f)
    return json_str

# Load image
image_path = "./static/goat.jpg"
image = imread(image_path)
width = image.shape[0] // 20
height = image.shape[0] // 20
gray_image = image[:, :, 2]

# Set page title and wide layout
st.title("SVD ( Singular Value Decomposition )", anchor="center")
st.write(
    """
An interactive demonstration of Singular Value Decomposition (SVD) applied to image reconstruction.
"""
)

common_markdown()

# Display the original image and reconstructed image side by side
margin = 0.1
_, col_original, spinner_col, col_reconstructed, _ = st.columns(
    (margin, 1, margin * 2, 1, margin), gap="large"
)

# Display original image
col_original.subheader("Original Image")
plt.figure(figsize=(width // 2, height // 2))
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.box(False)
plt.style.use("dark_background")
plt.imshow(image, cmap="gray")
col_original.pyplot(plt.gcf())

# Display rank selection
col_reconstructed.subheader("Reconstructed Image")
reconsructed_image = col_reconstructed.empty()
_, _, _, slider_area, _ = st.columns((margin, 1, margin * 2, 1, margin), gap="large")
# Reconstruct image using SVD
rank = slider_area.slider("Select Rank", min_value=1, max_value=200, value=50, on_change = reconsructed_image.empty)

with col_reconstructed:
    with st_lottie_spinner(load_json('./static/loading_animation.json'), reverse= True, quality= 'high', height = 300):
        st.toast("Decomposiing Image", icon="✅")
        U, S, VT = do_svd(gray_image)
        S = np.diag(S)
        st.toast("Reconstructeing Compressed Image", icon="⏳")
        Xprox = reconstruct(U, S, VT, rank)
        plt.figure(figsize=(width // 2, height // 2))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.box(False)
        plt.style.use("dark_background")
        plt.imshow(Xprox, cmap="gray")
        reconsructed_image.pyplot(plt.gcf())

        st.toast("Image Reconstructed", icon="✅")
