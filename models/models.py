import os, sys
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Removed wandb import as it's no longer needed
# import wandb

from utils.pytorch_ssim import SSIMLoss
from models.feature_extractor import Extractor


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        raise NotImplementedError

    def predict_anomaly(self, x: Tensor):
        raise NotImplementedError

    def load(self, path: str):
        """
        Load model from a local path.
        :param path: Path to the model file.
        """
        if not os.path.exists(path):
            raise RuntimeError(f"Model file {path} does not exist.")
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        print(f"Model loaded from {path}")

    def save(self, config, name: str, directory: str = "./models"):
        """
        Save model to a local directory.
        :param config: Configuration used for the model.
        :param name: Name of the model file.
        :param directory: Directory where the model will be saved.
        """
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, name)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': vars(config)  # Convert Namespace to dict
        }, save_path)
        print(f"Model saved to {save_path}")


def vanilla_feature_encoder(in_channels: int, hidden_dims: List[int],
                            norm_layer: str = None, dropout: float = 0.0,
                            bias: bool = False):
    """
    Vanilla feature encoder.
    Args:
        in_channels (int): Number of input channels
        hidden_dims (List[int]): List of hidden channel dimensions
        norm_layer (str): Normalization layer to use
        dropout (float): Dropout rate
        bias (bool): Whether to use bias
    Returns:
        encoder (nn.Module): The encoder
    """
    ks = 5  # Kernel size
    pad = ks // 2  # Padding

    # Build encoder
    enc = nn.Sequential()
    for i in range(len(hidden_dims)):
        # Add a new layer
        layer = nn.Sequential()

        # Convolution
        layer.add_module(f"encoder_conv_{i}",
                         nn.Conv2d(in_channels, hidden_dims[i], ks, stride=2,
                                   padding=pad, bias=bias))

        # Normalization
        if norm_layer is not None:
            layer.add_module(f"encoder_norm_{i}",
                             eval(norm_layer)(hidden_dims[i]))

        # LeakyReLU
        layer.add_module(f"encoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"encoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the encoder
        enc.add_module(f"encoder_layer_{i}", layer)

        in_channels = hidden_dims[i]

    # Final layer
    enc.add_module("encoder_conv_final",
                   nn.Conv2d(in_channels, in_channels, ks, stride=1, padding=pad,
                             bias=bias))

    return enc


def vanilla_feature_decoder(out_channels: int, hidden_dims: List[int],
                            norm_layer: str = None, dropout: float = 0.0,
                            bias: bool = False):
    """
    Vanilla feature decoder.
    Args:
        out_channels (int): Number of output channels
        hidden_dims (List[int]): List of hidden channel dimensions
        norm_layer (str): Normalization layer to use
        dropout (float): Dropout rate
        bias (bool): Whether to use bias
    Returns:
        decoder (nn.Module): The decoder
    """
    ks = 6  # Kernel size
    pad = 2  # Padding

    hidden_dims = [out_channels] + hidden_dims

    # Build decoder
    dec = nn.Sequential()
    for i in range(len(hidden_dims) - 1, 0, -1):
        # Add a new layer
        layer = nn.Sequential()

        # Transposed convolution
        layer.add_module(f"decoder_tconv_{i}",
                         nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1],
                                            ks, stride=2, padding=pad,
                                            bias=bias))

        # Normalization
        if norm_layer is not None:
            layer.add_module(f"decoder_norm_{i}",
                             eval(norm_layer)(hidden_dims[i - 1]))

        # LeakyReLU
        layer.add_module(f"decoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"decoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the decoder
        dec.add_module(f"decoder_layer_{i}", layer)

    # Final layer
    dec.add_module("decoder_conv_final",
                   nn.Conv2d(hidden_dims[0], out_channels, 1, bias=False))

    return dec


class FeatureAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.enc = vanilla_feature_encoder(config.in_channels,
                                           config.hidden_dims,
                                           norm_layer="nn.BatchNorm2d",
                                           dropout=config.dropout,
                                           bias=False)
        self.dec = vanilla_feature_decoder(config.in_channels,
                                           config.hidden_dims,
                                           norm_layer="nn.BatchNorm2d",
                                           dropout=config.dropout,
                                           bias=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.enc(x)
        rec = self.dec(z)
        return rec


class FeatureReconstructor(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.extractor = Extractor(inp_size=config.image_size,
                                   cnn_layers=config.extractor_cnn_layers,
                                   keep_feature_prop=config.keep_feature_prop,
                                   pretrained=not config.random_extractor)

        config.in_channels = self.extractor.c_feats
        self.ae = FeatureAutoencoder(config)

        if config.loss_fn == 'ssim':
            self.loss_fn = SSIMLoss(window_size=5, size_average=False)
        elif config.loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")

    def forward(self, x: Tensor):
        with torch.no_grad():
            feats = self.extractor(x)
        return feats, self.ae(feats)

    def get_feats(self, x: Tensor) -> Tensor:
        return self.extractor(x)

    def get_rec(self, feats: Tensor) -> Tensor:
        return self.ae(feats)

    def loss(self, x: Tensor):
        feats, rec = self(x)
        loss = self.loss_fn(rec, feats).mean()
        return {'rec_loss': loss}

    def predict_anomaly(self, x: Tensor):
        """Returns per image anomaly maps and anomaly scores"""
        # Extract features
        feats, rec = self(x)

        # Compute anomaly map
        anomaly_map = self.loss_fn(rec, feats).mean(1, keepdim=True)
        anomaly_map = F.interpolate(anomaly_map, x.shape[-2:], mode='bilinear',
                                    align_corners=True)

        # Anomaly score only where object in the image, i.e. at x > 0
        anomaly_score = []
        for i in range(x.shape[0]):
            roi = anomaly_map[i][x[i] > 0]
            if roi.numel() == 0:
                anomaly_score.append(torch.tensor(0.0, device=x.device))
                continue
            roi = roi[roi > torch.quantile(roi, 0.9)]
            if roi.numel() == 0:
                anomaly_score.append(torch.tensor(0.0, device=x.device))
            else:
                anomaly_score.append(roi.mean())
        anomaly_score = torch.stack(anomaly_score)
        return anomaly_map, anomaly_score


if __name__ == '__main__':
    # Config
    from argparse import Namespace
    config = Namespace()
    config.image_size = 128
    config.hidden_dims = [100, 150, 200, 300]
    config.generator_hidden_dims = [300, 200, 150, 100]
    config.discriminator_hidden_dims = [100, 150, 200, 300]
    config.dropout = 0.2
    config.extractor_cnn_layers = ['layer1', 'layer2']
    config.keep_feature_prop = 1.0
    config.random_extractor = False  # Added to match the extractor initialization
    config.loss_fn = 'ssim'  # Added to specify the loss function
    config.in_channels = 1  # Added to specify input channels
    device = "cpu"

    # Model
    fae = FeatureReconstructor(config).to(device)
    print("Encoder:")
    print(fae.ae.enc)
    print("\nDecoder:")
    print(fae.ae.dec)

    # Data
    x = torch.randn(32, config.in_channels, config.image_size, config.image_size).to(device)

    # Forward
    feats, rec = fae(x)
    print(f"Features shape: {feats.shape}, Reconstructed shape: {rec.shape}")

    anomaly_map, anomaly_score = fae.predict_anomaly(x)
    print(f"Anomaly map shape: {anomaly_map.shape}, Anomaly scores: {anomaly_score}")

    # Example of saving and loading the model
    # Save the model
    fae.save(config, "feature_reconstructor.pth", directory="./saved_models")

    # Load the model
    fae_loaded = FeatureReconstructor(config).to(device)
    fae_loaded.load("./saved_models/feature_reconstructor.pth")
    print("Model loaded successfully.")
