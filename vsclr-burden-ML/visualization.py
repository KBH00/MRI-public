import pydicom
import matplotlib.pyplot as plt

# Load the DICOM file
dicom_file_path = "D:/Download/FLAIR_T2_dataset/ADNI/003_S_4441/Accelerated_Sagittal_MPRAGE__MSV21_/2024-04-23_11_26_06.0/I10454353/ADNI_003_S_4441_MR_Accelerated_Sagittal_MPRAGE_(MSV21)__br_raw_20240509082626032_54.dcm"
dicom_data = pydicom.dcmread(dicom_file_path)

# Extract the pixel array (image data)
image_data = dicom_data.pixel_array

# Display the image using matplotlib
plt.imshow(image_data, cmap='gray')
plt.title("MRI Scan")
plt.axis('off')
plt.show()
