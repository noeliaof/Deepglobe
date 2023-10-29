import os
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
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


def main(config):


    ENCODER = config['ENCODER']
    ENCODER_WEIGHTS = config['ENCODER_WEIGHTS']

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    print('creating splits and datasets')
    train_df, valid_df, test_df = get_subset_data(config['DATA_DIR'], batch_size=3, num_workers=2, validation_split=0.1)
    select_class_rgb_values, class_rgb_values = get_classes(config['DATA_DIR'], config['CLASSES'])

    train_dataset = DeepGlobeDataset( train_df,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        one_hot_encode=one_hot_encode,
        class_rgb_values=select_class_rgb_values,
        )

    valid_dataset = DeepGlobeDataset( valid_df,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            one_hot_encode=one_hot_encode,
            class_rgb_values=select_class_rgb_values,
        )

    test_dataset = DeepGlobeDataset(test_df,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            one_hot_encode=one_hot_encode,
            class_rgb_values=select_class_rgb_values,
        )

    test_dataset_vis = DeepGlobeDataset(
            test_df,
            augmentation=get_validation_augmentation(),
            class_rgb_values=select_class_rgb_values,
        )

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH'], shuffle=True, num_workers=config['n_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=config['BATCH'], shuffle=False, num_workers=config['n_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH'], shuffle=False, num_workers=config['n_workers'])


    # build the model 
    print("building the model")
    model = build_model(config)

    print('training the model')
    train(model, train_loader, valid_loader, config['OUT_DIR'], num_epochs=config['EPOCHS'], learning_rate=config['lr'])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file for the model.')
    args = parser.parse_args()

    config = load_config(args.config)
    # Run the main function
    main(config)
