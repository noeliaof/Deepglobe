# in AWS Lambda, we need to use this import below
import os
import sys
import yaml
import torch
import boto3
import requests
import numpy as np
from urllib.request import urlopen
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
import base64



s3_resource = boto3.resource('s3')
bucketname = 'deepglobedata'
model_key = 'final_model.pth'
config_key = 'config.yaml'

def download_files(bucket=bucketname, files=[], destination='.'):
    downloaded_files = []
    s3_resource = boto3.resource('s3')

    for file_key in files:
        location = os.path.join(destination, os.path.basename(file_key))
        if not os.path.exists(location):
            s3_resource.Object(bucket, file_key).download_file(location)
            downloaded_files.append(location)
    
    return downloaded_files



# Define neccesary functions
def load_config(config_path):
    # Load configuration from file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def build_model(config):
    # Get the selected model class from the mapping in YAML
    model_classes = config['MODEL_CLASSES']
    selected_model_class = getattr(smp, model_classes[config['MODEL']])

    # Create segmentation model with pretrained encoder
    model = selected_model_class(
        encoder_name=config['ENCODER'],
        encoder_weights=config['ENCODER_WEIGHTS'],
        classes=len(config['CLASSES']),
        activation=config['ACTIVATION'],
    )

    return model

def load_model(config, model_path):
    # Load the model
    model = build_model(config)
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print("load state")
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        # Add additional information or logging here
        raise e
    print("movel eval")
    model.eval()
    return model


def infer_image(image):
    
    img = transforms.ToTensor()(image)
    print("shape", img.shape)
    # Add batch dimension
    img = img.unsqueeze(0)
    #img = img.to(DEVICE)
    with torch.no_grad():
        pred_mask = model(img)
        pred_mask = pred_mask.detach().cpu().numpy()
        pred_mask = pred_mask.squeeze()
   
    #true_pred = np.argmax(pred_mask,axis=0)
    print("before return infer_image")
    return pred_mask


def process_img(img_url):
    # load the image using PIL module
    img = Image.open(urlopen(img_url))
    return img


def get_classes(dir_metadata, classes):
    class_dict = pd.read_csv(os.path.join(dir_metadata, 'class_dict.csv'))
    # Get class names
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r','g','b']].values.tolist()

    print('All dataset classes and their corresponding RGB values in labels:')
    print('Class Names: ', class_names)
    print('Class RGB values: ', class_rgb_values)

    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    return  select_class_rgb_values, class_rgb_values

def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def decode_pred(pred_mask):

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
    return result



def lambda_handler(event, context):
    """upload the image and make the inference"""
    if 'body' in event and event['body']:
        # I assume the image is base64 encoded
        image_base64 = event['body']
        
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Download the model and configuration file to the current working directory
        downloaded_files = download_files(bucket=bucketname, files=[model_key, config_key], destination=os.getcwd())
        model_path = downloaded_files[0]  
        config_path = downloaded_files[1] 
        config = load_config(config_path)
        
        # Make prediction
        preds = infer_image(img)
        
        # Obtain the result
        results = decode_pred(preds)

        return results
    else:
        return {"error": "No image data provided in the event body"}
