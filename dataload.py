# fae/dataload.py

import os
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from skimage.transform import resize

class Dicom3DDataset(Dataset):
    def __init__(self, directories, transform=None, labels=None):
        """
        Custom dataset for loading 3D DICOM images.

        Args:
            directories (list): List of directories containing .dcm files.
            transform (callable, optional): Optional transform to be applied on a sample.
            labels (list, optional): Optional labels for supervised learning.
        """
        self.directories = directories
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, idx):
        directory = self.directories[idx]
        dcm_files = sorted([f for f in os.listdir(directory) if f.endswith(".dcm")])
        
        slices = []
        for dcm_file in dcm_files:
            file_path = os.path.join(directory, dcm_file)
            dicom_data = pydicom.dcmread(file_path)
            slice_image = dicom_data.pixel_array.astype(np.float32)  # Convert to float32

            if slice_image.ndim == 2:  
                slice_image_resized = resize(slice_image, (128, 128), anti_aliasing=True)
            elif slice_image.ndim == 3: 
                slice_image_resized = resize(slice_image, (128, 128, 128), anti_aliasing=True)
            else:
                raise ValueError(f"Unexpected slice dimensions: {slice_image.ndim} for file: {dcm_file}")

            slices.append(slice_image_resized)
        
        if len(slices) == 0:
            raise ValueError(f"No slices found in directory: {directory}")
        
        # Handle 2D and 3D slices differently
        if slices[0].ndim == 2:
            volume = np.stack(slices, axis=0)  # Shape: (num_slices, 128, 128)
        else:
            volume = slices[0]  # Assuming the first slice is already 3D

        # Convert to torch tensor and add channel dimension
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # Shape: (1, num_slices, 128, 128)
        
        # Resize to (1, num_slices, 128, 128, 128) if 3D
        if volume.ndim == 4:
            volume = F.interpolate(volume.unsqueeze(0), size=(128, 128, 128), mode='trilinear', align_corners=False)
            volume = volume.squeeze(0)  # Shape: (1, num_slices, 128, 128, 128)
        
        if self.transform:
            volume = self.transform(volume)
        label = self.labels[idx] if self.labels is not None else -1
        return volume, label

def find_dcm_directories(base_dir):
    """
    Recursively find all directories containing .dcm files.

    Args:
        base_dir (str): The base directory to search.

    Returns:
        list: List of directories containing at least one .dcm file.
    """
    dcm_directories = []
    for root, dirs, files in os.walk(base_dir):
        if any(file.endswith(".dcm") for file in files):
            dcm_directories.append(root)
    return dcm_directories

def get_dataloaders_csv(csv_path, train_base_dir, batch_size=4, transform=None, validation_split=0.2, seed=42):
    """
    Prepare and return DataLoaders for training, validation, and testing.

    Args:
        csv_path (str): Path to the CSV file containing 'path' and 'PXPERIPH' columns.
        train_base_dir (str): Base directory containing training DICOM directories.
        batch_size (int, optional): Batch size for DataLoaders. Default is 4.
        transform (callable, optional): Transformations to apply to the data. Default is Normalize((0.5,), (0.5,)).
        validation_split (float, optional): Fraction of test data to use for validation. Default is 0.5.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (train_loader, validation_loader, test_loader)
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),  
        ])

    data_labels = pd.read_csv(csv_path)

    label_map = {path: label for path, label in zip(data_labels['path'].dropna(), data_labels['PXPERIPH'])}
    test_directories = data_labels.loc[data_labels['path'].notna(), 'path'].tolist()

    expanded_test_directories = []
    expanded_labels = []
    for path, label in label_map.items():
        dcm_dirs = find_dcm_directories(path)

        if dcm_dirs:
            for dir in dcm_dirs:
                expanded_test_directories.append(dir)
                # Map labels: 1 -> 0 (normal), 2 -> 1 (abnormal)
                if label == 1:
                    mapped_label = 0
                elif label == 2:
                    mapped_label = 1
                else:
                    raise ValueError(f"Unknown label {label} for path {path}")
                expanded_labels.append(mapped_label)

    if len(expanded_test_directories) != len(expanded_labels):
        print(f"Warning: Number of expanded directories ({len(expanded_test_directories)}) does not match number of labels ({len(expanded_labels)})")

    test_dataset_full = Dicom3DDataset(expanded_test_directories, transform=transform, labels=expanded_labels)

    validation_size = int(len(test_dataset_full) * validation_split)
    test_size = len(test_dataset_full) - validation_size

    torch.manual_seed(seed)

    validation_dataset, test_dataset = random_split(test_dataset_full, [validation_size, test_size])

    train_directories = find_dcm_directories(train_base_dir)
    train_dataset = Dicom3DDataset(train_directories, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("="*30)
    print("Training DataLoader:")
    for batch in train_loader:
        volumes, labels = batch
        print(f"Batch volume shape: {volumes.shape}, Batch label shape: {labels.shape}")
        break  
    print("="*30)
    print("Validation DataLoader:")
    for batch in validation_loader:
        volumes, labels = batch
        print(f"Batch volume shape: {volumes.shape}, Batch label shape: {labels.shape}")
        break  
    print("="*30)
    print("Test DataLoader:")
    for batch in test_loader:
        volumes, labels = batch
        print(f"Batch volume shape: {volumes.shape}, Batch label shape: {labels.shape}")
        break  
    return train_loader, validation_loader, test_loader

def get_dataloaders(csv_path, train_base_dir, batch_size=4, transform=None, validation_split=0.1, test_split=0.1, seed=42):
    """
    Prepare and return DataLoaders for training, validation, and testing.

    Args:
        csv_path (str): Path to the CSV file containing 'path' and 'PXPERIPH' columns.
        train_base_dir (str): Base directory containing training DICOM directories.
        batch_size (int, optional): Batch size for DataLoaders. Default is 4.
        transform (callable, optional): Transformations to apply to the data. Default is Normalize((0.5,), (0.5,)).
        validation_split (float, optional): Fraction of the dataset to use for validation. Default is 0.2.
        test_split (float, optional): Fraction of the dataset to use for testing. Default is 0.1.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (train_loader, validation_loader, test_loader)
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),  
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        ])
    torch.manual_seed(seed)

    print("Data load....")

    train_directories = find_dcm_directories(train_base_dir)
    train_dataset = Dicom3DDataset(train_directories, transform=transform)

    total_size = len(train_dataset)
    validation_size = int(total_size * validation_split)
    test_size = int(total_size * test_split)
    train_size = total_size - validation_size - test_size
    print(validation_split, train_size, validation_size, test_size)

    train_dataset, validation_dataset, test_dataset = random_split(train_dataset, [train_size, validation_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("Data Loader ready...")
    
    return train_loader, validation_loader, test_loader

