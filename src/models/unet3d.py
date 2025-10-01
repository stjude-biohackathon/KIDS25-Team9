import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import tifffile as tiff
from abc import ABC, abstractmethod
import base_model

#PARAMETERS
IMAGE_DIR = 'Images'  # Directory containing input images
LABEL_DIR = 'Labels'  # Directory containing ground truth masks
BATCH_SIZE = 1  # Smaller batch size for 3D due to memory constraints
NUM_EPOCHS = 30 # Number of training epochs
LEARNING_RATE = 1e-4 # Learning rate for optimizer
DATASET_SPLIT = [0.5, 0.5]  # Fraction of data to use for training
THRESHOLD = 0.5  # Threshold for converting probabilities to binary masks


class Unet3DNapari(base_model.BaseModel):
    def __init__(self):
        super().__init__()
        # Initialize 3D U-Net specific parameters here
        self.model_name = "3D U-Net"
        # Add more attributes as needed

    @abstractmethod
    def architecture(self):
        pass

    @abstractmethod
    def train(self, data):
        # Check the device we are using is GPU or CPU
        pass


    @abstractmethod
    def infer(self, input_data):
        pass


class DoubleConv3D(nn.Module):
    """Double 3D convolution block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation"""

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Encoder (downsampling path)
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Decoder (upsampling path)
        for feature in reversed(features):
            # Transposed convolution for upsampling
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # Double convolution after concatenation
            self.ups.append(DoubleConv3D(feature * 2, feature))

        # Final 1x1x1 convolution
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool3d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections.reverse()

        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip_connection = skip_connections[idx // 2]

            # Handle potential size mismatches
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            # Concatenate skip connection
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # Double convolution

        return self.final_conv(x)


class CustomDataset3D(Dataset):
    """Dataset for loading 3D volumetric data"""

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.volumes = os.listdir(img_dir)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, item):
        vol_path = os.path.join(self.img_dir, self.volumes[item])
        mask_path = os.path.join(self.mask_dir, self.volumes[item])

        # Load volumetric data from .tif
        volume = np.array(tiff.imread(vol_path))
        mask = np.array(tiff.imread(mask_path))

        volume = self.transform(volume)
        mask = self.transform(mask)
        mask = (mask > 0).float()  # Ensure mask is binary (0 or 1)
        # volume = np.ascontiguousarray(volume)
        # mask = np.ascontiguousarray(mask)
        # volume = torch.from_numpy(volume).float()
        # mask = torch.from_numpy(mask).float()

        # Add channel dimension if needed
        if volume.ndim == 3:
            volume = np.expand_dims(volume, axis=0)
        if mask.ndim == 3:
            mask = np.expand_dims(mask, axis=0)

        volume = torch.from_numpy(volume).float()
        mask = torch.from_numpy(mask).float()

        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        mask = (mask > 0).float()

        return volume, mask


def train_3d(model, num_epochs, train_loader, optimizer, loss_function, device, print_every=10):
    """Training loop for 3D U-Net"""
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for count, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            out = model(x)
            out = torch.sigmoid(out)

            # Compute loss
            loss = loss_function(out, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if count % print_every == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{count}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}')


def eval_3d(model, val_loader, device):
    """Evaluation function for 3D U-Net"""
    model.eval()
    num_correct = 0
    num_voxels = 0
    total_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            out = model(x)
            probability = torch.sigmoid(out)
            predictions = probability > THRESHOLD

            # Calculate accuracy
            num_correct += (predictions == y).sum().item()
            num_voxels += y.numel()

    accuracy = num_correct / num_voxels
    print(f'Validation Accuracy: {accuracy:.4f}')
    return accuracy


# Example usage
if __name__ == "__main__":
    # Configuration
    # BATCH_SIZE = 1  # Smaller batch size for 3D due to memory constraints
    # NUM_EPOCHS = 30
    # DEPTH = 16  # D dimension
    # IMG_HEIGHT = 128  # H dimension
    # IMG_WIDTH = 128  # W dimension
    # LEARNING_RATE = 1e-4

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = UNet3D(in_channels=1, out_channels=1, features=[32, 64, 128, 256])
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load your 3D dataset
    all_data = CustomDataset3D(IMAGE_DIR, LABEL_DIR, T.Compose([T.ToTensor(), ]))
    train_data, val_data = torch.utils.data.random_split(all_data, DATASET_SPLIT)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Training
    train_3d(model, NUM_EPOCHS, train_loader, optimizer, loss_function, device)
    eval_3d(model, val_loader, device)

    # Save model
    torch.save(model.state_dict(), 'unet3d_model.pth')

    # Inference and save predicted masks
    model.eval()
    with torch.no_grad():
        for idx, (x, _) in enumerate(val_loader):
            x = x.to(device)
            out = model(x)
            pred_mask = (torch.sigmoid(out) > THRESHOLD).cpu().numpy()
            # print(f'Predicted mask shape: {pred_mask.shape}')
            # # Save each mask in the batch
            # # for i in range(pred_mask.shape[0]):
            # #     tiff.imwrite(f'segmentation_{idx}_{i}.tif', pred_mask[i, 0])
            # tiff.imwrite(f'segmentation_{idx}.tif', pred_mask)
            original_mask = pred_mask[0, 0]  # Remove batch and channel
            # If needed, transpose axes to match original (depth, height, width)
            # For example, if your original was (height, width, depth):
            original_mask = np.transpose(original_mask, (1, 2, 0))
            tiff.imwrite('segmentation.tif', original_mask)