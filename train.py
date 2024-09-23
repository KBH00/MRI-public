import torch
import torch.optim as optim
import torch.nn as nn
from model import SpineNet
from preprocessing import train_loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SpineNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device).float()  # Convert to float
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            # print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
            # print(f"Outputs: {outputs}, Labels: {labels}")

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    print('Finished Training')

train_model(model, train_loader, criterion, optimizer, num_epochs=10)
