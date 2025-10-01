import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utilities.unet_2d_dataset_builder import SegmentationDataset
from unet_2d import UNet
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import numpy
import json
from pathlib import Path

def main():   
    # Load config from JSON
    with open("../unet_2d_config.json", "r") as f:
        args = json.load(f)

    # Unpack config into variables
    mode            = args.get("mode", "train")
    image_dir       = args["image_dir"]
    mask_dir        = args["mask_dir"]
    num_channels    = args["channels"]
    num_classes     = args["classes"]
    weights_path    = args["weights_path"]
    output_dir      = args["output_dir"]
    epochs          = args.get("epochs", 20)
    batch_size      = args.get("batch_size", 2)
    learning_rate   = args.get("learning_rate", 1e-3)

    # Init model
    model = UNet(num_channels, num_classes)

    if mode == "architecture":
        model.architecture()
        return

    if mode == "train":
        dataset = SegmentationDataset(image_dir, mask_dir)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), learning_rate)

        train_data = {
            "dataloader": dataloader,
            "optimizer": optimizer,
            "criterion": criterion,
            "device": device,
            "epochs": epochs
        }
        model.train(train_data)

        torch.save(model.state_dict(), weights_path)
        print(f"Training complete. Model saved to weights_path")
    
    if mode == "infer":
        dataset = SegmentationDataset(image_dir, mask_dir)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i, (images, masks) in enumerate(dataloader):
            infer_data = {
                "weights_path": weights_path,
                "images": images,
                "device": device
            }
            infer_masks = model.infer(infer_data)

            #mask_np = infer_masks[0].cpu().numpy().astype("uint8") * 255
            mask_np = infer_masks[0].cpu().numpy().astype("uint16") * 65535
            out_path = Path(output_dir) / f"infer_mask_{i}.tif"
            Image.fromarray(mask_np).save(out_path)
            print(f"Saved {out_path}, unique values: {numpy.unique(mask_np)}")


if __name__ == "__main__":
    main()