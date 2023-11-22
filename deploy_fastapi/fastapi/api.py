import asyncio
import io
import pdb
import sys
import io

import cv2
import numpy as np
import uvicorn
from PIL import Image
from starlette.responses import Response

from fastapi import FastAPI, File, UploadFile

from utils import get_classes,colour_code_segmentation, reverse_one_hot
from helpers import infer_image, load_config
from helpers import config


app = FastAPI(
    title="DeepLabV3 image segmentation")


@app.get("/")
def read_root():
    return {"message": "Welcome to the root of the API"}

# Your other routes and functions go here

@app.post("/segment")
def segment(file: UploadFile):

    content = file.file.read()
    image = Image.open(io.BytesIO(content))
    # Perform segmentation using the model
    pred_mask = infer_image(image)
    
    print("shape of predictions")
    print(pred_mask.shape)
    select_class_rgb_values, class_rgb_values = get_classes(config['DATA_DIR'], config['CLASSES'])
    palette = np.array(class_rgb_values, dtype=np.uint8).flatten()
    
    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    #pred_urban_land_heatmap = pred_mask[:, :, config['CLASSES'].index('urban_land')]
    pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)

    #pdb.set_trace()
    #output.putpalette(palette)
    # Display the segmented image using st.pyplot
    result = Image.fromarray(np.uint8(pred_mask))
    bytes_io = io.BytesIO()
    result.save(bytes_io, format="PNG")

    return Response(bytes_io.getvalue(), media_type="image/png")


if __name__ == "__main__":
    # Use uvicorn to run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)

