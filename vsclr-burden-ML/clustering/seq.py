import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

# Load images in batches from preprocessed .npy files
def load_images_in_batches(preprocessed_data_dir, batch_size=100):
    image_files = [f for f in os.listdir(preprocessed_data_dir) if f.endswith(".npy")]
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        images = [np.load(os.path.join(preprocessed_data_dir, f)) for f in batch_files]
        yield np.array(images), batch_files  # Return batch files to track filenames

preprocessed_data_dir = "D:/Download/Preprocessed_MRI"
kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=100)

all_reduced_data = []
all_clusters = []

def visualize_progress(all_reduced_data, all_clusters, batch_num):
    pca_final = PCA(n_components=2)
    reduced_data_2d = pca_final.fit_transform(np.vstack(all_reduced_data))
    
    plt.clf()  
    plt.scatter(reduced_data_2d[:, 0], reduced_data_2d[:, 1], c=np.concatenate(all_clusters), cmap='viridis', s=5, alpha=0.5)
    plt.title(f"Clustering Progress after Batch {batch_num + 1}")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.pause(0.1)  # Pause to update the plot

plt.figure(figsize=(10, 8))

for batch_num, (batch, batch_files) in enumerate(load_images_in_batches(preprocessed_data_dir, batch_size=100)):
    batch = batch.reshape(len(batch), -1)
    
    # Perform PCA to reduce dimensionality to 50 components
    pca = PCA(n_components=50)
    reduced_batch = pca.fit_transform(batch)
    
    # Fit MiniBatchKMeans on the batch
    kmeans.partial_fit(reduced_batch)
    
    # Predict cluster labels for the batch
    batch_clusters = kmeans.predict(reduced_batch)
    
    all_reduced_data.append(reduced_batch)
    all_clusters.append(batch_clusters)
    
    visualize_progress(all_reduced_data, all_clusters, batch_num)
    print(f"Processed batch {batch_num + 1}")

# Show the final plot
plt.show()
print("Clustering completed!")
