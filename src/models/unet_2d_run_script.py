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
    with open("../configs/unet_2d/unet_2d_config.json", "r") as f:
        args = json.load(f)    

    # Init model
    num_channels = args["channels"]
    num_classes = args["classes"]
    model = UNet(num_channels, num_classes)

    if args["mode"] == "architecture":
        model.architecture()
        return

    if args["mode"] == "train":
        dataset = SegmentationDataset(args["image_dir"], args["mask_dir"])
        dataloader = DataLoader(dataset, batch_size=args.get("batch_size", 2), shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.get("learning_rate", 1e-3))

        train_data = {
            "dataloader": dataloader,
            "optimizer": optimizer,
            "criterion": criterion,
            "device": device,
            "epochs": args.get("epochs", 20)
        }
        model.train(train_data)

        torch.save(model.state_dict(), args["weights_path"])
        print(f"Training complete. Model saved to {args['weights_path']}")
    
    if args["mode"] == "infer":
        dataset = SegmentationDataset(args["image_dir"], args["mask_dir"])
        dataloader = DataLoader(dataset, batch_size=args.get("batch_size", 2), shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i, (images, masks) in enumerate(dataloader):
            infer_data = {
                "weights_path": args["weights_path"],
                "images": images,
                "device": device
            }
            infer_masks = model.infer(infer_data)

            #mask_np = infer_masks[0].cpu().numpy().astype("uint8") * 255
            mask_np = infer_masks[0].cpu().numpy().astype("uint16") * 65535
            out_path = Path(args["output_dir"]) / f"infer_mask_{i}.tif"
            Image.fromarray(mask_np).save(out_path)
            print(f"Saved {out_path}, unique values: {numpy.unique(mask_np)}")


if __name__ == "__main__":
    main()