import streamlit as st
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf


# Load the model
def load_model():
    model = tf.keras.models.load_model("my_model.h5")
    return model


model = load_model()

# import base64
#
#
# def get_image(image_file):
#     with open(image_file, "rb") as img_file:
#         encoded = base64.b64encode(img_file.read()).decode()
#     return encoded
# #
# # #
# image = get_image("TvsA.jpg")
#
# def home():
#     st.markdown(
#         f"""<style>
#     .stApp {{
#         background-image: url("data:image/jpg;base64,{image}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }}
#     </style>""", unsafe_allow_html=True
#     )

import streamlit as st

import streamlit as st

# Set page config
st.set_page_config(page_title="Background Image", layout="wide")

# Background Image URL (Replace with your own image URL)
background_image_url = "https://daijiworld.ap-south-1.linodeobjects.com/Linode/images3/spl_301023_1.jpg"  # Replace with your image URL

# Background Image CSS for Full Page and Sidebar
page_bg_img = f"""
<style>
    /* Main App Background */
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Sidebar Background */
    [data-testid="stSidebar"] {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Example content
# st.sidebar.title("Sidebar Menu")
# st.sidebar.write("This sidebar also has a background image.")
#
# st.title("Streamlit with Full Background Image")
# st.write("This example shows how to add a background image to both the main content area and the sidebar.")




def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "Details"])

    if page == "Home":
        show_home()
    elif page == "Prediction":
        show_prediction()
    elif page == "Details":
        show_details()


def show_home():
    st.markdown("<h1 style='text-align:center;'>APPLE OR TOMATO</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='font-weight:1000;'>This project focuses on developing a machine learning model to classify images of apples and tomatoes.Using a comprehensive dataset containing images of both fruits, the model is trained to recognize and distinguish between the two based on visual features like shape, color, and texture. The project showcases the power of computer vision techniques to accurately classify visually similar objects, with potential applications in agriculture, retail, and inventory management. Through this, we demonstrate the efficiency of AIin automating image recognition tasks for precise categorization and analysis.</p>", unsafe_allow_html=True)
    # st.write("**This project focuses on developing a machine learning model to classify images of apples and tomatoes.Using a comprehensive dataset containing images of both fruits, the model is trained to recognize and distinguish between the two based on visual features like shape, color, and texture. The project showcases the power of computer vision techniques to accurately classify visually similar objects, with potential applications in agriculture, retail, and inventory management. Through this, we demonstrate the efficiency of AIin automating image recognition tasks for precise categorization and analysis**")

    # st.write("""
    #         This project focuses on developing a machine learning model to classify images of apples and tomatoes.
    #         Using a comprehensive dataset containing images of both fruits, the model is trained to recognize and
    #         distinguish between the two based on visual features like shape, color, and texture. The project showcases
    #         the power of computer vision techniques to accurately classify visually similar objects, with potential
    #         applications in agriculture, retail, and inventory management. Through this, we demonstrate the efficiency of AI
    #         in automating image recognition tasks for precise categorization and analysis.
    #         """)


def show_prediction():
    st.markdown("<h2 style='text-align:center;'>Make a Prediction</h2>", unsafe_allow_html=True)
    image = st.file_uploader("Choose an image to predict...", type=["jpg", "jpeg", "png"])

    if image:
        # Read and preprocess the image
        image = imread(image)
        image = resize(image, (150, 150, 3))  # Resize to match model's expected input
        image = image.reshape(1, 150, 150, 3)  # Reshape for the model
        st.image(image,caption="Uploaded Image",width=400)


        # Make prediction
        y_new = model.predict(image)
        ind = y_new.argmax()

        if ind.any() == 0:
            st.write("Prediction: *APPLE*")
        else:
            st.write("Prediction: *TOMATO*")


def show_details():
    st.markdown("<h2 style='text-align:center;'>Model Details</h2>", unsafe_allow_html=True)
    # st.write("""
    # The model is a convolutional neural network (CNN) trained on a dataset of images to classify individuals as either drowsy or natural.
    # It utilizes various layers to extract features and make accurate predictions.
    # - *Input Layer*: Takes images resized to 150x150 pixels.
    # - *Convolutional Layers*: Extract features from images.
    # - *Pooling Layers*: Reduce dimensionality.
    # - *Dense Layers*: Final classification.
    #
    # Ensure to upload clear images for better prediction results.
    # """)

    st.markdown("<h3>References</h3>", unsafe_allow_html=True)
    st.markdown(
        "[Link to google colab](https://colab.research.google.com/drive/1Cl0mW73qqV4MUwskLLECjmzHY1jVjgE3#scrollTo=EoFtiTl0I4x1)",
        unsafe_allow_html=True)
    st.markdown("[Link to Dataset](https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset)",
                unsafe_allow_html=True)


main()
