# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel

class UNetDown(nn.Module):
    def __init__(self, input_size, output_size):
        super(UNetDown, self).__init__()
        
        model = [nn.BatchNorm2d(input_size),
                 nn.ELU(),
                 nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(output_size),
                 nn.ELU(),
                 nn.MaxPool2d(2),
                 nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):        
        return self.model(x)
      

class UNetUp(nn.Module):
    def __init__(self, input_size, output_size):
        super(UNetUp, self).__init__()

        model = [nn.BatchNorm2d(input_size),
                 nn.ELU(),
                 nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
                 nn.BatchNorm2d(output_size),
                 nn.ELU(),
                 nn.Upsample(scale_factor=2, mode="nearest"),
                 nn.Conv2d(output_size, output_size, kernel_size=3, stride=1, padding=1)]
          
        self.model = nn.Sequential(*model)
            
    def forward(self, x):
        return self.model(x)
            
         
class UNet(nn.Module, BaseModel):
    def __init__(self, channels_in, channels_out=2):
        super(UNet, self).__init__()

        self.conv_in = nn.Conv2d(
            channels_in, 64,
            kernel_size=3, stride=1, padding=1
        )  # [B,64,H,W]

        self.down1 = UNetDown(64, 64)      # [B,64,H/2,W/2]
        self.down2 = UNetDown(64, 128)     # [B,128,H/4,W/4]
        self.down3 = UNetDown(128, 128)    # [B,128,H/8,W/8]
        self.down4 = UNetDown(128, 256)    # [B,256,H/16,W/16]

        self.up4 = UNetUp(256, 128)        # [B,128,H/8,W/8]
        self.up5 = UNetUp(128 * 2, 128)    # [B,128,H/4,W/4]
        self.up6 = UNetUp(128 * 2, 64)     # [B,64,H/2,W/2]
        self.up7 = UNetUp(64 * 2, 64)      # [B,64,H,W]

        self.conv_out = nn.Conv2d(
            64 * 2, channels_out,
            kernel_size=3, stride=1, padding=1
        )  # [B,Co,H,W]

    def forward(self, x):
        x0 = self.conv_in(x)   # [B,64,H,W]

        x1 = self.down1(x0)    # [B,64,H/2,W/2]
        x2 = self.down2(x1)    # [B,128,H/4,W/4]
        x3 = self.down3(x2)    # [B,128,H/8,W/8]
        x4 = self.down4(x3)    # [B,256,H/16,W/16]

        # bottleneck: x4

        # upsample + skip connections
        x5 = self.up4(x4)                # [B,128,H/8,W/8]
        x5 = self._crop_and_concat(x5, x3)

        x6 = self.up5(x5)                # [B,128,H/4,W/4]
        x6 = self._crop_and_concat(x6, x2)

        x7 = self.up6(x6)                # [B,64,H/2,W/2]
        x7 = self._crop_and_concat(x7, x1)

        x8 = self.up7(x7)                # [B,64,H,W]
        x8 = self._crop_and_concat(x8, x0)

        out = self.conv_out(F.elu(x8))
        return out

    @staticmethod
    def _crop_and_concat(up, down):
        """
        Make sure skip connection tensors match in H,W before concatenation.
        If shapes differ by 1 pixel, pad the smaller one.
        """
        diffY = down.size()[2] - up.size()[2]
        diffX = down.size()[3] - up.size()[3]

        up = F.pad(
            up,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        return torch.cat([up, down], dim=1)

    # Implement BaseModel methods
    def architecture(self):
        print("2D UNet architecture with skip connections.")
        print(self)
        return self

    def train(self, data=True):
        if isinstance(data, bool):
            return super().train(mode=data)  # call nn.Module.train(mode)
        
        dataloader = data["dataloader"]
        optimizer = data["optimizer"]
        criterion = data["criterion"]
        device = data["device"]
        epochs = data["epochs"]
        
        self.to(device)
        super().train(mode=True)
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0.0
            total_pixels = 0
            #for imgs, masks in dataloader:
            for batch_idx, (imgs, masks) in enumerate(dataloader):
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = self.forward(imgs)

                #print(f"outputs.shape: {outputs.shape}, masks.shape: {masks.shape}")
                #print("mask unique:", masks.unique())

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                #if batch_idx == 0:
                    #print("First batch loss:", loss.item())
                # Track accuracy
                preds = torch.argmax(outputs, dim=1)  # [B,H,W]
                correct = (preds == masks).float().sum().item()
                total = masks.numel()
                epoch_acc += correct
                total_pixels += total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {epoch_acc / total_pixels:.4f}")

    #def infer(self, weights_path, input_tensor, device):
    def infer(self, input_data):
        weights_path = input_data["weights_path"]
        images = input_data["images"]
        device = input_data["device"]
        self.load_state_dict(torch.load(weights_path, map_location=device))
        self.to(device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(images.to(device))
            return torch.argmax(logits, dim=1)
            
