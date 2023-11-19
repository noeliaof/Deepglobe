import os
import torch
import albumentations as album
import segmentation_models_pytorch as smp
from utils import *
import segmentation_models_pytorch as smp


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