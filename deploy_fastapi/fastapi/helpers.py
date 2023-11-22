import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tempfile
from PIL import Image
from torchvision import transforms
from utils import *
from model import load_config, load_model


model_path = 'models/final_model.pth'
config_path = 'config.yaml'
config = load_config(config_path)
model= load_model(config, model_path)

def infer_image(image):
    
    img = transforms.ToTensor()(image)
    print("shape", img.shape)
    # Add batch dimension
    img = img.unsqueeze(0)
    pred_mask = model(img).detach().cpu().numpy()
    pred_mask = pred_mask.squeeze()
   
    #true_pred = np.argmax(pred_mask,axis=0)
    print("before return infer_image")
    return pred_mask



def image_to_rgb(predicted_image_array):
  predicted_image_array = np.transpose(predicted_image_array, (1, 2, 0))
  return Image.fromarray(np.uint8(predicted_image_array.argmax(axis=-1)))


def vis_patch(image_vis):
    """Visualize single patch and save it as an image"""
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    axs.imshow(image_vis)
    axs.set_title('Image')

    # Save the plot as a temporary image file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile, format='png', bbox_inches='tight')
        img_path = tmpfile.name

    plt.close()  # Close the plot to release resources

    return img_path  # Return the path to the saved image


def visualize_mask(pred_mask, config, gt_mask=None):
    select_class_rgb_values, class_rgb_values = get_classes(config['DATA_DIR'], config['CLASSES'])

    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    pred_urban_land_heatmap = pred_mask[:, :, config['CLASSES'].index('urban_land')]
    pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)

    fig, axs = plt.subplots(1, 1, figsize=(15, 5))

    axs.imshow(pred_mask)
    axs.set_title('Predicted mask')

    legend_patches = [mpatches.Patch(color=np.array([c]) / 255, label=class_name)
                      for class_name, c in zip(config['CLASSES'], class_rgb_values)]
    axs.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Save the plot as a temporary image file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile, format='png', bbox_inches='tight')
        img_path = tmpfile.name

    plt.close()  # Close the plot to release resources

    return img_path
