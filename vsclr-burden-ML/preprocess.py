import os
import pydicom
import numpy as np
from tqdm import tqdm
#import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt

data_dir = "D:/Download/FLAIR_T2_dataset/ADNI"
output_dir = "D:/Download/Preprocessed_MRI"

os.makedirs(output_dir, exist_ok=True)

def load_dicom_image(dicom_file_path):
    dicom_data = pydicom.dcmread(dicom_file_path)
    image = dicom_data.pixel_array.astype(np.float32)
    
    # Normalize pixel values between 0 and 1
    image -= np.min(image)
    image /= np.max(image)
    
    return image

def preprocess_image(image, target_size=(256, 256)):
    # Resize image to target size
    resized_image = resize(image, target_size, anti_aliasing=True)

    # Normalize the image
    resized_image -= np.mean(resized_image)
    resized_image /= np.std(resized_image)

    return resized_image

def load_preprocess_and_save(data_dir, output_dir, target_size=(256, 256)):
    image_counter = 0
    
    for root, dirs, files in tqdm(os.walk(data_dir)):
        print(root)
        for file in files:
            continue
            if file.endswith(".dcm"):
                dicom_file_path = os.path.join(root, file)
                
                try:
                    # Load and preprocess the image
                    image = load_dicom_image(dicom_file_path)
                    preprocessed_image = preprocess_image(image, target_size)
                    
                    # Save each image as a .npy file (or .png, as you prefer)
                    save_path = os.path.join(output_dir, f"image_{image_counter}.npy")
                    np.save(save_path, preprocessed_image)
                    
                    image_counter += 1
                except Exception as e:
                    print(f"Error processing {dicom_file_path}: {e}")
                    
    print(f"Preprocessed and saved {image_counter} images.")

load_preprocess_and_save(data_dir, output_dir, target_size=(256, 256))
