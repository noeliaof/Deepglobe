from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Dataset import DeepGlobeDataset 
import os
import numpy as np
import torch
from utils.utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils_scores import *
from utils.utils import *

def train(model, train_loader, valid_loader, save_dir, num_epochs, learning_rate, plot=True):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    # define loss function
    loss = smp.utils.losses.DiceLoss()

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # define optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=learning_rate),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # Lists to store training statistics for plotting
    train_loss_history = []
    train_iou_history = []
    valid_loss_history = []
    valid_iou_history = []
    
    print(num_epochs)
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.to(DEVICE)
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        # Create a progress bar for the training loop
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (inputs, target_masks) in progress_bar:
            inputs = inputs.to(DEVICE)
            target_masks = target_masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = loss(outputs, target_masks)
            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.item()

            # Calculate IoU for training
            outputs_bin = (outputs > 0.5).float()
            iou_batch = (outputs_bin * target_masks).sum() / ((outputs_bin + target_masks) > 0).sum()
            train_iou += iou_batch.item()

        # Calculate average loss and IoU for training
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_iou = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, target_masks) in enumerate(valid_loader):
                inputs = inputs.to(DEVICE)
                target_masks = target_masks.to(DEVICE)

                outputs = model(inputs)
                loss_value = loss(outputs, target_masks)

                valid_loss += loss_value.item()

                # Calculate IoU for validation
                outputs_bin = (outputs > 0.5).float()
                iou_batch = (outputs_bin * target_masks).sum() / ((outputs_bin + target_masks) > 0).sum()
                valid_iou += iou_batch.item()

        # Calculate average loss and IoU for validation
        valid_loss /= len(valid_loader)
        valid_iou /= len(valid_loader)

        # Store training statistics for plotting
        train_loss_history.append(train_loss)
        train_iou_history.append(train_iou)
        valid_loss_history.append(valid_loss)
        valid_iou_history.append(valid_iou)

      
        if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, "final_model_0.pth"))

    if plot:
      # Plot training statistics
      plt.figure(figsize=(12, 4))
      plt.subplot(1, 2, 1)
      plt.plot(range(1, epoch+2), train_loss_history, label='Train Loss')
      plt.plot(range(1, epoch+2), valid_loss_history, label='Valid Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(range(1, epoch+2), train_iou_history, label='Train IoU')
      plt.plot(range(1, epoch+2), valid_iou_history, label='Valid IoU')
      plt.xlabel('Epoch')
      plt.ylabel('IoU')
      plt.legend()

      plt.show()
