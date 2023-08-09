import streamlit as st
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Load image
image_path = "sample_image.jpg"
image = imread(image_path)
width = image.shape[0]//500
height = image.shape[0]//500
gray_image = image[:, :, 2]

# Set page title and wide layout
st.set_page_config(layout="wide")
st.title("SVD ( Singular Value Decomposition )", anchor="center")
st.write("""
An interactive demonstration of Singular Value Decomposition (SVD) applied to image reconstruction.
""")

# Sidebar description
st.sidebar.write("""
## Description
This app showcases how Singular Value Decomposition (SVD) can be used to reconstruct images. Use the slider to select the rank and observe the changes in the reconstructed image.
""")

# Display the original image and reconstructed image side by side
margin = 0.1
_,col_original,_,col_reconstructed,_ = st.columns((margin,1,margin,1,margin), gap = "large")

# Display original image
col_original.subheader("Original Image")
plt.figure(figsize = (width//2,height//2))
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.box(False)
plt.style.use('dark_background')           
plt.imshow(image ,cmap = 'gray')
col_original.pyplot(plt.gcf())

@st.cache_data(show_spinner = False)
def do_svd(x):
    U, S, VT = np.linalg.svd(x, full_matrices=False)
    return U, S, VT

@st.cache_data(show_spinner = False)
def reconstruct(U,S,VT,rank):
    Xprox = U[:, :rank] @ S[:rank, :rank] @ VT[:rank, :]
    return Xprox

U, S, VT = do_svd(gray_image)
S = np.diag(S)

# Display rank selection
col_reconstructed.subheader("Reconstructed Image")
_,_,_,slider_area,_ = st.columns((margin,1,margin,1,margin), gap = "large")
# Reconstruct image using SVD
rank = slider_area.slider("Select Rank", min_value=1, max_value=200, value=50)
with col_reconstructed:
    with st.spinner('Preparing Image'):
        Xprox = reconstruct(U,S,VT,rank)
        plt.figure(figsize = (width//2,height//2))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.box(False)
        plt.style.use('dark_background')
        plt.imshow(Xprox, cmap = 'gray')
        col_reconstructed.pyplot(plt.gcf())

# Add qtips
st.sidebar.markdown("#### What is Singular Value Decomposition?")
st.sidebar.markdown("""
## Singular Value Decomposition (SVD)
Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three other matrices:
                    $$A = U \Sigma V^T$$
- $$U$$ - the left singular vectors matrix
- $$\Sigma$$ - a diagonal matrix containing the singular values
- $$V^T$$ - the transpose of the right singular vectors matrix

SVD is a powerful tool used in various applications, such as image compression and data analysis.
""")

st.sidebar.markdown("#### How to Use?")
st.sidebar.markdown("1. Observe the original image on the left.")
st.sidebar.markdown("2. Use the slider to select the rank for reconstruction.")
st.sidebar.markdown("3. The reconstructed image will be displayed on the right, showing the effect of reducing the rank.")

st.sidebar.markdown("#### Try It!")
st.sidebar.markdown("Feel free to adjust the slider and explore how different rank values affect the reconstructed image.")
st.sidebar.markdown("#### Here is the Code")
st.sidebar.markdown(' The code begins by importing necessary libraries and loading the sample image. It then sets up the Streamlit page configuration and introduces the purpose of the app.')
st.sidebar.code(
    '''
    import streamlit as st
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

# Load image
image_path = "sample_image.jpg"
image = imread(image_path)
width = image.shape[0]//500
height = image.shape[0]//500
gray_image = image[:, :, 2]

# Set page title and wide layout
st.set_page_config(page_title="SVD Image Reconstruction", layout="wide")
st.title("", anchor="center")
st.write("""
# Singular Value Decomposition Demo
An interactive demonstration of Singular Value Decomposition (SVD) applied to image reconstruction.
""")

# Sidebar description
st.sidebar.write("""
## Description
This app showcases how Singular Value Decomposition (SVD) can be used to reconstruct images. Use the slider to select the rank and observe the changes in the reconstructed image.
""")

# Display the original image and reconstructed image side by side
margin = 0.3
_,col_original,_,col_reconstructed,_ = st.columns((margin,1,margin,1,margin), gap = "large")

# Display original image
col_original.subheader("Original Image")
plt.figure(figsize = (width//2,height//2))
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.box(False)
plt.style.use('dark_background')           
plt.imshow(image)
col_original.pyplot(plt.gcf())
''')

st.sidebar.markdown('Here, the code defines a function for caching the SVD computation to optimize performance. It then performs SVD on the grayscale image and constructs a diagonal matrix from the singular values.')

st.sidebar.code('''
@st.cache_data(show_spinner = False)
def do_svd(x):
    U, S, VT = np.linalg.svd(x, full_matrices=False)
    return U, S, VT

@st.cache_data(show_spinner = False)
def reconstruct(U,S,VT,rank):
    Xprox = U[:, :rank] @ S[:rank, :rank] @ VT[:rank, :]
    return Xprox

U, S, VT = do_svd(gray_image)
S = np.diag(S)''')

st.sidebar.markdown('This part sets up the section for displaying the reconstructed image. It defines a column layout for the reconstructed image and adds a slider to select the rank for image reconstruction.')
st.sidebar.code('''
# Display rank selection
col_reconstructed.subheader("Reconstructed Image")
_,_,_,slider_area,_ = st.columns((margin,1,margin,1,margin), gap = "large")
# Reconstruct image using SVD
rank = slider_area.slider("Select Rank", min_value=1, max_value=200, value=50)
with col_reconstructed:
    with st.spinner('Preparing Image'):
        Xprox = reconstruct(U,S,VT,rank)
        plt.figure(figsize = (width//2,height//2))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.box(False)
        plt.style.use('dark_background')
        plt.imshow(Xprox)
        col_reconstructed.pyplot(plt.gcf())
    '''
)

