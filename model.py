import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics

from PIL import Image, ImageFile
from Dataset import DeepGlobeDataset
from utils.utils import *
import segmentation_models_pytorch as smp
import yaml


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


def load_model(config):
    model = build_model()
    model_path = os.path.join(config['OUT_DIR'], "final_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model