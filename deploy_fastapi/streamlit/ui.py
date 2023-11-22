import os
import streamlit as st
from streamlit_image_comparison import image_comparison
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import numpy as np
from PIL import Image
import logging
from datetime import datetime

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

import io

from datetime import datetime



# FastAPI endpoint
DOMAIN = os.getenv("FASTAPI_DOMAIN", "localhost")
url = f"http://{DOMAIN}:8000"
endpoint = '/segment'

st.set_page_config(layout="wide")

st.sidebar.info(
    """
    - Web App URL: to be added
    - GitHub repository: to be added
    """
)

st.sidebar.title("Models")  # Move the model selection to the left sidebar
# Define a list of available models
available_models = ["DeepLabV3"]


# Model selection dropdown
selected_model = st.sidebar.selectbox(
   "Please select one model",
   ("DeepLabV3", "Unet++", "other"),
   index=None,
   placeholder="Select contact method...",
)

# Check if the selected model is available
if selected_model != "":
    option = selected_model
    st.sidebar.write(f"Selected value: {option}")
elif selected_model == "":
    st.sidebar.warning("Please choose a model from the dropdown.")


# Customize page title
st.title('Image segmentation')


st.write('''Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
         This Streamlit example uses a FastAPI service as a backend.
         Visit this URL at `:8000/docs` for FastAPI documentation.''')  # Description and instructions

st.header("Instructions")


image_file = st.file_uploader('Upload image')  # image upload widge
#print(image_file)
if image_file is not None:
    image = Image.open(image_file)
    #displaying the image on streamlit app
    st.image(image, caption='Enter any caption here',  width=300)

def process(image, server_url: str):
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    m = MultipartEncoder(
        fields={'file': ('filename', img_byte_arr, 'image/jpeg')}
        )

    print("Data being sent in the request:")
    #print(m.fields)
    try:
        logging.info("Request started")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        r = requests.post(server_url, data=m, headers={'Content-Type': m.content_type}, timeout=1000)
        logging.info("Request completed")
        return r
    except Exception as e:
        print("An error occurred during the POST request:")
        print(str(e))
        raise e



if st.button('Get segmentation map'):

    if image_file == None:
        st.write("Insert an image!")  # handle case with no image
    else:
        segments = process(image, url+endpoint)
        segmented_image = Image.open(io.BytesIO(segments.content)).convert('RGB')
       
        # Image comparison
        image_comparison(
        img1=image,#image_to_rgb(satellite_input_imageobj),#satellite_input_imageobj,
        img2=segmented_image,#image_to_rgb(predicted_output_imageobj),#model_predict_result),
        label1="Original Satellite Image",
        label2="Predicted Label Image",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
                