import os
import sys
import argparse
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.models import FeatureReconstructor
from utils.pytorch_ssim import SSIMLoss
from models.feature_extractor import Extractor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x  # If tqdm is not installed, use a dummy loop


class InferenceDataset(Dataset):
    def __init__(self, directories: List[str], transform=None):
        """
        Dataset for inference.

        Args:
            directories (List[str]): List of directories containing .dcm files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directories = directories
        self.transform = transform

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
        
        if slices[0].ndim == 2:
            volume = np.stack(slices, axis=0)  # Shape: (num_slices, 128, 128)
        else:
            volume = slices[0]  

        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # Shape: (1, num_slices, 128, 128)
        
        # Resize to (1, num_slices, 128, 128, 128) if 3D
        if volume.ndim == 4:
            volume = F.interpolate(volume.unsqueeze(0), size=(128, 128, 128), mode='trilinear', align_corners=False)
            volume = volume.squeeze(0)  # Shape: (1, num_slices, 128, 128, 128)
        
        if self.transform:
            volume = self.transform(volume)
            
        return volume, directory  


def find_dcm_directories(base_dir: str) -> List[str]:
    """
    Recursively find all directories containing .dcm files.

    Args:
        base_dir (str): The base directory to search.

    Returns:
        List[str]: List of directories containing at least one .dcm file.
    """
    dcm_directories = []
    for root, dirs, files in os.walk(base_dir):
        if any(file.endswith(".dcm") for file in files):
            dcm_directories.append(root)
    return dcm_directories


def visualize_anomaly_overlay(original_slice: torch.Tensor, anomaly_map: torch.Tensor, save_path: str = None):
    """
    Visualize the overlay of the anomaly map on the original MR image.

    Args:
        original_slice (torch.Tensor): The original MR image slice of shape (H, W).
        anomaly_map (torch.Tensor): The anomaly map of shape (H, W).
        save_path (str, optional): Path to save the visualization. If None, displays the plot.
    """

    original_slice = original_slice.squeeze().cpu().numpy()
    anomaly_map = anomaly_map.squeeze().cpu().numpy()

    plt.figure(figsize=(6, 6))
    
    plt.imshow(original_slice, cmap='gray', interpolation='none')
    plt.imshow(anomaly_map, cmap='hot', alpha=0.5, interpolation='none')
    
    plt.colorbar()
    plt.title('MR Image with Anomaly Map Overlay')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Anomaly map overlay saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Inference with Feature Autoencoder')
    parser.add_argument('--checkpoint', type=str, default="./saved_models/model_epoch_5.pth", help='Path to the model checkpoint file')
    parser.add_argument('--input_dir', type=str, default="D:/Data/FLAIR_T2_ss/ADNI/", help='Directory containing DICOM directories for inference')
    parser.add_argument('--output_dir', type=str, default='./anomaly_maps', help='Directory to save anomaly map visualizations')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from argparse import Namespace
    config = Namespace()
    config.image_size = 128
    config.hidden_dims = [100, 150, 200, 300]
    config.generator_hidden_dims = [300, 200, 150, 100]
    config.discriminator_hidden_dims = [100, 150, 200, 300]
    config.dropout = 0.2
    config.extractor_cnn_layers = ['layer1', 'layer2']
    config.keep_feature_prop = 1.0
    config.random_extractor = False
    config.loss_fn = 'mse'

    model = FeatureReconstructor(config).to(args.device)
    
    model.load(args.checkpoint)
    model.eval()

    dcm_directories = find_dcm_directories(args.input_dir)
    if len(dcm_directories) == 0:
        print(f"No DICOM directories found in {args.input_dir}")
        return

    inference_dataset = InferenceDataset(dcm_directories)
    inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for batch_idx, (volumes, directories) in enumerate(tqdm(inference_loader, desc="Inference")):
            volumes = volumes.to(args.device)  
            B, C, D, H, W = volumes.shape

            # Flatten the slices into (B*D, 1, H, W) for per-slice anomaly detection
            volumes_slices = volumes.view(B * D, C, H, W)  # Shape: (B*D, 1, H, W)

            # Predict anomaly for each slice
            anomaly_map, anomaly_score = model.predict_anomaly(volumes_slices)  # anomaly_map: (B*D, 1, H, W)
            anomaly_map = anomaly_map.view(B, D, 1, H, W)  # Reshape back to (B, D, 1, H, W)

            # For visualization, select the middle slice of each volume
            middle_slice = D // 2
            for i in range(B):
                dir_path = directories[i]
                volume_id = os.path.basename(dir_path.rstrip('/\\'))
                path_id = dir_path.split('/')[-1].replace('\\','.')

                # Extract the original MR slice and corresponding anomaly map
                original_slice = volumes[i, 0, middle_slice, :, :]  # Shape: (H, W)
                single_anomaly_map = anomaly_map[i, middle_slice, 0, :, :]  # Shape: (H, W)

                save_path = os.path.join(args.output_dir, f"{path_id}.png")
                visualize_anomaly_overlay(original_slice, single_anomaly_map, save_path=save_path)

                print(f"Volume: {volume_id}, Anomaly Score: {anomaly_score[i].item():.4f}")

    print("Inference completed.")


if __name__ == '__main__':
    main()
