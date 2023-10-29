from fastapi import File, UploadFile, Request, FastAPI
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import base64


import pdb

import io
from apis import *

app = FastAPI()
templates = Jinja2Templates(directory="templates")

#imports 
# Define the paths to the model and config
model_path = '/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/models/final_model.pth'
config_path = '/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/config.yaml'



@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
  
@app.post("/upload")
def upload(request: Request, file: UploadFile = File(...)):
    try:
        image = cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
        #contents = file.file.read()
        patch_vis = vis_patch(image)
        with open(patch_vis, "rb") as img_file:
            contents = img_file.read()
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
        
    base64_encoded_image = base64.b64encode(contents).decode("utf-8")

    return templates.TemplateResponse("display.html", {"request": request,  "myImage": base64_encoded_image})





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





