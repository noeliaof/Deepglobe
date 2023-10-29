import os
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics 

from PIL import Image, ImageFile
from Dataset import DeepGlobeDataset
from utils.utils import *
from utils.get_subset_data import *
from train import *
from model import *
import yaml
import argparse


import torch
import numpy as np
from PIL import Image
from utils.viz import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def preprocess_image(image_path):
    # Add any necessary preprocessing steps for your input image
    # For example, resizing, normalization, etc.
    img = Image.open(image_path)
    # Perform any necessary transformations on 'img'
    # ...

    # Convert the image to a PyTorch tensor
    img = transforms.ToTensor()(img)

    # Add batch dimension
    img = img.unsqueeze(0)
    return img

def inference_single_image(model, image_path):

    model.eval()  # Set the model to evaluation mode

    # Preprocess the input image
    input_image = preprocess_image(image_path)

    # Move the input to the device used for training (e.g., GPU)
    input_image = input_image.to(DEVICE)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)

    # Process the output as needed
    # (e.g., applying softmax for classification)
    # processed_output = ...

    return output.cpu().numpy()




if __name__ == "__main__":
    # Load the trained model

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='directory of images')

    parser.add_argument('--config', type=str, default='config.yaml', help='config file for the model.')

    parser.add_argument('--model_dir', type=str, default='/Users/noeliaotero/Documents/WeCloudData/Capstone_project/Deepglobe/models/final_model.pth',
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")
    
    args = parser.parse_args()
    config = load_config(args.config)
    trained_model_path = args.model_dir
    
    print("loading the model")
    model = build_model(config)
    state_dict = torch.load(trained_model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)

    # Specify the path to the input image
    input_image_path = args.data_dir

    examples = os.listdir(args.data_dir)
    image_files = [os.path.join(args.data_dir, filename) for filename in examples if filename.endswith('.jpg')]
    
    # Perform inference on a single image
    print("testing one image", image_files[0])
    result = inference_single_image(model, image_files[0])
    result = result.squeeze()
    print(result.shape)

    print('visualize results')
    input_img = preprocess_image(image_files[0])
    input_img = input_img.squeeze()
    
    visualize_predictions(np.transpose(input_img, (1, 2, 0)),  result, config, gt_mask =None)
