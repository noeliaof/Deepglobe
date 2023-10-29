import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Dataset import DeepGlobeDataset  # Replace with the correct module
import os
import numpy as np
from utils import *

def get_subset_data(DATA_DIR, batch_size=3, num_workers=0, validation_split=0.1):


    metadata = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
    metadata_df = metadata[metadata['split']=='train']
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    # Shuffle DataFrame
    total_samples = len(metadata_df)
    train_size = int(0.7 * total_samples)
    test_size = int(0.2 * total_samples)
    #valid_size = total_samples - train_size - test_size

    # Split the DataFrame into train, test, and valid subsets
    train_df = metadata_df.iloc[:train_size]
    test_df = metadata_df.iloc[train_size:train_size + test_size]
    valid_df = metadata_df.iloc[train_size + test_size:]

    # Reset the indices for each subset
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)


    return train_df, valid_df, test_df


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