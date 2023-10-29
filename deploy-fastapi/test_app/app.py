import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import pdb

import io
from apis import *

app = FastAPI()

# Mount the "static" directory to serve frontend files
app.mount("/static", StaticFiles(directory="/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/data/test/"), name="static")


#@app.get("/")
#def hello():
#    return {"message" : "Welcome to segment test"}

# Define the paths to the model and config
model_path = '/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/models/final_model.pth'
config_path = '/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/config.yaml'

# Create an instance of the SegmentationModel class
#segmentation_model = SegmentationModel(model_path, config_path)

# Serve the HTML file at the root URL
@app.get("/")
async def get_index():
    html_file = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_file)


@app.post("/upload")
async def vis_image(file: UploadFile):
    # Process the uploaded image
    image = cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    patch_vis = vis_patch(image)
    with open(patch_vis, "rb") as img_file:
        image_bytes = img_file.read()
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@app.post("/segment")
async def segment_image(file: UploadFile):
    try:
        # Process the uploaded image
        image = cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
       # image = np.transpose(image, (2, 0, 1))
       # print(image.shape)
       # pdb.set_trace()
        # Perform segmentation using the model
        result = infer_image(config_path, model_path, image)
       # result = segmentation_model.segment_image(image)
        #result = result.squeeze()
        
        print('visualize results')
        #input_img = preprocess_image(result)
        config = load_config(config_path)
        #pdb.set_trace()
        processed_image = visualize_mask(result, config, gt_mask=None)
        # Save the processed image to a file
       # processed_image_path = "/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/deploy-fastapi/test.jpg"  # Set the path where you want to save the processed image
       # cv2.imwrite(processed_image_path, processed_image)
        with open(processed_image, "rb") as img_file:
            image_bytes = img_file.read()        
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
    except Exception as e:
        return {"error": str(e)}



