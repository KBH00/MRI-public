import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

# Assuming preprocessed_images_tensor is the tensor output from your preprocessing pipeline
# Example shape: (N, 1, 256, 256) where N is the number of images

class MRIDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]  # Binary: 0 = Normal, 1 = Vascular burden

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transforms for Data Augmentation and normalization
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to the size expected by ResNet50
    transforms.Normalize([0.485], [0.229]),  # Normalize with ImageNet mean and std
    transforms.RandomHorizontalFlip(),  # Data Augmentation
    transforms.RandomRotation(10),      # Data Augmentation
])

# Create dataset
labels = np.random.randint(0, 2, size=len(preprocessed_images_tensor))  # Random labels (you should have your actual labels)
dataset = MRIDataset(preprocessed_images_tensor, labels, transform=data_transforms)

# Create DataLoader
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# Modify the first convolutional layer to accept 1 channel instead of 3
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify the fully connected layer to output 2 classes (normal, vascular burden)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print('Training complete')

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
