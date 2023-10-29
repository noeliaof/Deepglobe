import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io

st.title('DeepLabV3 image segmentation')

# FastAPI endpoint
url = "http://localhost:8000"
endpoint = '/segment'

st.write('''Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
         This Streamlit example uses a FastAPI service as a backend.
         Visit this URL at `:8000/docs` for FastAPI documentation.''')  # Description and instructions

#uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])  # Image upload widget

#if uploaded_image is not None:
#    # Read the image data from the file uploader
#    image_data = uploaded_image.read()

#    m = MultipartEncoder(
#        fields={"file": ("image.jpg", image_data, "image/jpeg")}
#    )
#    response = requests.post(url, data=m, headers={'Content-Type': m.content_type}, timeout=8000)

    # Display the segmented image from FastAPI
#    st.image(response.content, caption="Segmented Image", use_column_width=True)


image_file = st.file_uploader('Upload image')  # image upload widge
print(image_file)
if image_file is not None:
    image = Image.open(image_file)
    #displaying the image on streamlit app
    st.image(image, caption='Enter any caption here')


def process(image, server_url: str):
    print(f"Processing image with server_url: {server_url}")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    m = MultipartEncoder(
        fields={'file': ('filename', img_byte_arr, 'image/jpeg')}
        )

    print("Data being sent in the request:")
    print(m.fields)
    try:
        r = requests.post(server_url, data=m, headers={'Content-Type': m.content_type}, timeout=8000)
    except Exception as e:
        print("An error occurred during the POST request:")
        print(str(e))

    return r



if st.button('Get segmentation map'):

    if image_file == None:
        st.write("Insert an image!")  # handle case with no image
    else:
        segments = process(image, url+endpoint)
        print(segments)
        print("Response content:", segments.content)
        # Image.fromarray(np.uint8(pred_mask))
        segmented_image = Image.open(io.BytesIO(segments.content)).convert('RGB')
        st.image([image_file, segmented_image], width=300)  # output dyptich