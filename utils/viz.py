import os
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
from train import *
from model import *
from utils.get_subset_data import *
import yaml
import argparse


def visualize_predictions(image_vis, pred_mask, config, gt_mask=None):

    select_class_rgb_values, class_rgb_values = get_classes(config['DATA_DIR'], config['CLASSES'])

    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    # Get prediction channel corresponding to foreground
    pred_urban_land_heatmap = pred_mask[:, :, config['CLASSES'].index('urban_land')]
    pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)

    fig, axs = plt.subplots(1, 2 + int(gt_mask is not None), figsize=(15, 5))

    axs[0].imshow(image_vis)
    axs[0].set_title('Image')

    axs[1].imshow(pred_mask)
    axs[1].set_title('Predicted mask')

    if gt_mask is not None:
        # Convert gt_mask from `CHW` format to `HWC` format
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        gt_mask = colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values)
        axs[2].imshow(gt_mask)
        axs[2].set_title('Ground Truth')

        # Create legend
        legend_patches = [mpatches.Patch(color=np.array([c]) / 255, label=class_name)
                          for class_name, c in zip(config['CLASSES'], class_rgb_values)]

        axs[2].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.show()



def vis_patch(image_vis):
    """Visualize single patch"""
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))

    axs[0].imshow(image_vis)
    axs[0].set_title('Image')

    plt.show()


def visualize_mask(pred_mask, config, gt_mask=None):
    """Visualize single mask and predicted mask with optional legend"""
    select_class_rgb_values, class_rgb_values = get_classes(config['DATA_DIR'], config['CLASSES'])

    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    # Get prediction channel corresponding to foreground
    pred_urban_land_heatmap = pred_mask[:, :, config['CLASSES'].index('urban_land')]
    pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
    
    plt.imshow(pred_mask)
    #fig, axs = plt.subplots(1, 1 + int(gt_mask is not None), figsize=(15, 5))
   # fig, axs = plt.subplots(1, 1, figsize=(15, 5))
   # axs.imshow(pred_mask)
   # axs.set_title('Predicted mask')

    #if gt_mask is not None:
    #    # Convert gt_mask from `CHW` format to `HWC` format
    #    gt_mask = np.transpose(gt_mask, (1, 2, 0))
    #    gt_mask = colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values)
    #    axs[1].imshow(gt_mask)
    #    axs[1].set_title('Ground Truth')
   
    # Create legend
    #legend_patches = [mpatches.Patch(color=np.array([c]) / 255, label=class_name)
    #                      for class_name, c in zip(config['CLASSES'], class_rgb_values)]
    #axs[0].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    #plt.show()
