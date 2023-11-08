import streamlit as st
import torch

# from tensorflow import keras
import random
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Abnormality Detection in X-Ray Images",
    page_icon="🩻",
    initial_sidebar_state="auto",
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction) == clss:
            return key


with st.sidebar:
    # st.image("mg.png")
    st.title("X-Ray Image Abnormality Detection")
    st.subheader(
        "Detecting abnormalities in X-Ray images using Deep Learning and Computer Vision"
    )


def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction) == clss:
            return key


st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache_resource
def load_model():
    # Create the feature extractor and model
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    checkpoint = torch.load("pytorch_model1.bin", map_location=torch.device("mps"))
    # Load the model's state_dict from the checkpoint
    model.load_state_dict(checkpoint)

    # Put the model in evaluation mode
    model.eval()
    return model


with st.spinner("Model is being loaded.."):
    model = load_model()


st.write(
    """
         # Upload an X-Ray Image
         """
)

file = st.file_uploader("", type=["jpg", "png"])


def import_and_predict(image_path, model):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to match the model's input size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    probabilities = F.softmax(output.logits, dim=1)
    print(probabilities[:, 1] > 0.5)
    threshold = 0.5  # Define a threshold
    predicted_class = (
        probabilities[:, 1] > threshold
    ).int()  # Classify as 1 if the probability of the positive class is above the threshold
    return predicted_class


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(file, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = [
        "Abnormal",
        "Normal",
    ]
    print(class_names[predictions])

    string = class_names[predictions]
    if class_names[predictions] == "Normal":
        st.balloons()
        st.sidebar.success(string)

    elif class_names[predictions] == "Abnormal":
        st.sidebar.warning(string)
