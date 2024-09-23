import os
import pandas as pd
import pydicom
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Load CSV files
train_df = pd.read_csv('./Data/train.csv')
train_coords_df = pd.read_csv('./Data/train_label_coordinates.csv')

# Merge train_df with train_coords_df to get series_id, condition, level, and coordinates
train_merged_df = pd.merge(train_df, train_coords_df, on="study_id")

# Specify condition columns that indicate labels
condition_columns = [col for col in train_df.columns if 'spinal_canal_stenosis' in col or 'neural_foraminal_narrowing' in col or 'subarticular_stenosis' in col]

# Severity mapping
severity_mapping = {
    "Normal/Mild": 0,
    "Moderate": 1,
    "Severe": 2
}

# Map severity levels to numeric values
for col in condition_columns:
    train_merged_df[col] = train_merged_df[col].map(severity_mapping)

train_merged_df[condition_columns] = train_merged_df[condition_columns].fillna(0)  # Handle missing values

# Custom Dataset Class
class SpineMRIDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.condition_columns = [col for col in dataframe.columns if 'spinal_canal_stenosis' in col or 'neural_foraminal_narrowing' in col or 'subarticular_stenosis' in col]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        img_path = os.path.join(".", "Data", self.image_dir, str(row['study_id']), str(row['series_id']), f"{row['instance_number']}.dcm")
        dicom = pydicom.dcmread(img_path)
        img_array = dicom.pixel_array

        # Normalize the image and resize
        img = img_array / 255.0
        img = cv2.resize(img, (256, 256))  

        if self.transform:
            img = self.transform(img)

        # Use the condition columns to get the labels
        labels = row[self.condition_columns].values.astype(np.float32)

        return img, torch.tensor(labels)

# Image transformation with data augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

# Create Dataset and DataLoader
train_dataset = SpineMRIDataset(train_merged_df, 'train_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load evaluation dataset CSV file
eval_df = pd.read_csv('./Data/eval.csv')  # Replace with your evaluation CSV
eval_dataset = SpineMRIDataset(eval_df, 'eval_images', transform=eval_transform)

# DataLoader for evaluation
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)