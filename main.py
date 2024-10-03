# fae/main.py

import argparse
from dataload import get_dataloaders
from models.models import FeatureReconstructor
from utils.pytorch_ssim import SSIMLoss  
import torch
import torch.nn as nn
import torch.optim as optim
import os

#D:/VascularData/data/ADNI
#D:/Data/FLAIR_T2_ss/ADNI
def parse_args():
    parser = argparse.ArgumentParser(description='Train Feature Autoencoder on 3D DICOM Images')
    parser.add_argument('--csv_path', type=str, default="C:/Users/kbh/Desktop/CNI/test/updated_subject_paths.csv", help='Path to the CSV file containing DICOM paths and labels')
    parser.add_argument('--train_base_dir', type=str, default="D:/VascularData/data/ADNI", help='Base directory for training DICOM files')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for DataLoaders')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    return parser.parse_args()

def main():
    args = parse_args()

    train_loader, validation_loader, test_loader = get_dataloaders(
        csv_path=args.csv_path,
        train_base_dir=args.train_base_dir,
        batch_size=args.batch_size,
        transform=None,  
        validation_split=0.1,
        seed=42
    )

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
    config.loss_fn = 'ssim'
    # config.in_channels will be set by FeatureReconstructor based on Extractor

    model = FeatureReconstructor(config).to(args.device)

    print("Feature Autoencoder Encoder:")
    print(model.ae.enc)
    print("\nFeature Autoencoder Decoder:")
    print(model.ae.dec)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (volumes, _) in enumerate(train_loader):
            volumes = volumes.to(args.device)  # Shape: (B, 1, D, H, W)
            B, C, D, H, W = volumes.shape

            volumes_slices = volumes.view(B * D, C, H, W)  # Shape: (B*D, 1, H, W)

            # Forward pass
            # feats, rec = model(volumes_slices)  # feats: (B*D, C_feats, H', W'), rec: same
            # loss_dict = model.loss(feats)
            loss_dict = model.loss(volumes_slices)

            loss = loss_dict['rec_loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx +1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{args.epochs}], Average Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for volumes, _ in validation_loader:
                volumes = volumes.to(args.device)
                B, C, D, H, W = volumes.shape
                volumes_slices = volumes.view(B * D, C, H, W)

                loss_dict = model.loss(volumes_slices)

                loss = loss_dict['rec_loss']

                val_loss += loss.item()

        val_loss /= len(validation_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        scheduler.step()

        if epoch%5==0:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pth")
            model.save(config, f"model_epoch_{epoch}.pth", directory=args.save_dir)
            print(f"Model saved to {save_path}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for volumes, _ in test_loader:
            volumes = volumes.to(args.device)
            B, C, D, H, W = volumes.shape
            volumes_slices = volumes.view(B * D, C, H, W)

            loss_dict = model.loss(volumes_slices)

            loss = loss_dict['rec_loss']

            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    final_model_path = os.path.join(args.save_dir, "model_final.pth")
    model.save(config, "model_final.pth", directory=args.save_dir)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    main()
