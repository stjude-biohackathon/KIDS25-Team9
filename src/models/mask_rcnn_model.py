from abc import ABC, abstractmethod
import os
import json
import numpy as np
from skimage import io
import datetime
import torch
import torchvision
from torchvision.transforms import functional as F
from base_model import BaseModel
from utilities.maskrcnn_dataset_builder import CellDataset, build_dataset_dataloader
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class MaskRCNNModel(BaseModel):

    def __init__(self, num_classes=2, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.architecture(num_classes).to(self.device)

    def architecture(self, num_classes=2):
        # Load pretrained Mask R-CNN
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")

        # Replace box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )

        # Replace mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        return model

    def train(self, train_loader, val_loader = None, learning_rate = 0.001,saved_dir = 'checkpints',  num_epochs=4):
        self.model.train()
        # set up optimizer
        self.optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate, momentum=0.9, weight_decay=learning_rate*0.1
        )

        # Create timestamped checkpoint folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(saved_dir, f"maskrcnn_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {save_dir}")

        best_val_dice = float("-inf")
        best_model_path = None

        for epoch in range(1, num_epochs + 1):
            
            # Training loop
            self.model.train()
            total_loss = 0
            for images, targets in train_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                total_loss += losses.item()
            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
            # Validation
            if val_loader:
                mean_iou, mean_dice = self.validate_metrics(val_loader)
                # Save best model by Dice score
                if mean_dice > best_val_dice:   
                    best_val_dice = mean_dice
                    best_model_path = os.path.join(save_dir, f"maskrcnn_best_epoch{epoch}.pth")
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"  🔥 New best model (Dice {best_val_dice:.4f}) saved at {best_model_path}")
                

            # Save every 5 epochs
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f"maskrcnn_epoch{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  ✅ Saved checkpoint at {checkpoint_path}")
        # Save final model
        final_path = os.path.join(save_dir, f"maskrcnn_final_epoch{num_epochs}.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"Final model saved at {final_path}")
        return

    def infer(self, model_path, image_path, score_thresh=0.5):

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        img = io.imread(image_path)
        # ----- Preprocess image -----
        if isinstance(img, np.ndarray):
            # Convert grayscale -> RGB
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            # Normalize image to [0,1] float32
            img = img.astype(np.float32)
            if img.max() > 0:
                img /= img.max()
            img = F.to_tensor(img)  # [3,H,W]
        else:
            raise TypeError("Input must be numpy array or torch tensor")
        img = img.to(self.device)
        # ----- Inference -----
        with torch.no_grad():
            preds = self.model([img])
        # ----- Postprocess -----
        scores = preds[0]["scores"].cpu().numpy()
        keep = scores >= score_thresh
        result = {
            "boxes": preds[0]["boxes"].cpu().numpy()[keep],
            "masks": preds[0]["masks"].squeeze(1).cpu().numpy()[keep] > 0.5,  # [N,H,W]
            "labels": preds[0]["labels"].cpu().numpy()[keep],
            "scores": scores[keep]}
        return result
        

    def save_inference(self, model_path, image_path, score_thresh=0.5, save_path=None):
        # Run inference
        result = self.infer(model_path=model_path, image_path=image_path, score_thresh=score_thresh)
        # Load original image for visualization
        img = io.imread(image_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img.astype(np.float32)
        if img.max() > 1.0:  # normalize if not already
            img /= img.max()
        # Prepare save path
        if save_path is None:
            base, _ = os.path.splitext(image_path)
            save_path = f"{base}_inference.png"
            save_mask_path = f"{base}_mask.png"
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img, cmap="gray")

        boxes = result["boxes"]
        masks = result["masks"]
        labels = result["labels"]
        scores = result["scores"]
        # Overlay masks with random colors
        overlay = np.zeros_like(img, dtype=np.float32)
        for i, mask in enumerate(masks):
            color = np.random.rand(3)
            overlay[mask] = color
            xmin, ymin, xmax, ymax = boxes[i]
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                xmin, ymin - 5,
                f"{labels[i]}:{scores[i]:.2f}",
                color="yellow", fontsize=8, weight="bold"
            )
        # Blend masks with original image
        blended = (0.7 * img + 0.3 * overlay).clip(0, 1)
        ax.imshow(blended)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"✅ Saved inference visualization at {save_path}")
        plt.imsave(save_mask_path, overlay)
        print(f"✅ Saved mask overlay at {save_mask_path}")
        return save_path

    def infer_mask(self, model_path, input_path, output_path, score_thresh=0.5, num_classes=6):
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        # Load image
        img = io.imread(input_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img.astype(np.float32)
        if img.max() > 0:
            img /= img.max()
        img_tensor = F.to_tensor(img).to(self.device)

        # Inference
        with torch.no_grad():
            preds = self.model([img_tensor])
        scores = preds[0]["scores"].cpu().numpy()
        keep = scores >= score_thresh
        masks = preds[0]["masks"].squeeze(1).cpu().numpy()[keep] > 0.5
        labels = preds[0]["labels"].cpu().numpy()[keep]

        # Prepare output folders
        os.makedirs(output_path, exist_ok=True)
        image_id = os.path.splitext(os.path.basename(input_path))[0]
        h, w = img.shape[:2]

        # For each class, make subfolder and save one mask
        for cls in range(1, num_classes):  # skip background=0
            class_dir = os.path.join(output_path, str(cls))
            os.makedirs(class_dir, exist_ok=True)

            class_mask = np.zeros((h, w), dtype=np.uint16)
            inst_id = 1
            for mask, lbl in zip(masks, labels):
                if lbl == cls:
                    class_mask[mask] = inst_id
                    inst_id += 1

            out_path = os.path.join(class_dir, f"{image_id}.tif")
            io.imsave(out_path, class_mask.astype(np.uint16), check_contrast=False)
        print(f"✅ Saved predictions for {image_id} at {output_path}")
        return output_path
        
    def infer_mask_batch(self, model_path, input_dir, output_dir, score_thresh=0.5, num_classes=6):
        img_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".tif")])
        for img_file in img_files:
            input_path = os.path.join(input_dir, img_file)
            image_id = os.path.splitext(img_file)[0]
            self.infer_mask(model_path, input_path, output_dir, score_thresh, num_classes)
        
            


    def compute_iou(self, pred_mask, true_mask):
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return intersection / union

    def compute_dice(self, pred_mask, true_mask):
        """Compute Dice score between two binary masks."""
        intersection = np.logical_and(pred_mask, true_mask).sum()
        total = pred_mask.sum() + true_mask.sum()
        if total == 0:
            return 1.0
        return 2 * intersection / total
    
    def validate_metrics(self, val_loader, score_thresh=0.5):
        """
        Run validation using IoU and Dice instead of loss.
        Returns mean IoU and mean Dice across validation set.
        """
        self.model.eval()
        iou_scores = []
        dice_scores = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)  # predictions only
                for output, target in zip(outputs, targets):
                    pred_masks = (output["masks"] > 0.5).squeeze().cpu().numpy()
                    true_masks = target["masks"].cpu().numpy()
                    # If no predictions, skip
                    if pred_masks.ndim == 2:
                        pred_masks = pred_masks[None, ...]  # ensure [N,H,W]
                    if true_masks.ndim == 2:
                        true_masks = true_masks[None, ...]

                    # Compare each GT mask with best matching pred
                    for tmask in true_masks:
                        best_iou, best_dice = 0.0, 0.0
                        for pmask in pred_masks:
                            iou = self.compute_iou(pmask, tmask)
                            dice = self.compute_dice(pmask, tmask)
                            if iou > best_iou:
                                best_iou = iou
                            if dice > best_dice:
                                best_dice = dice
                        iou_scores.append(best_iou)
                        dice_scores.append(best_dice)

        mean_iou = np.mean(iou_scores) if iou_scores else 0.0
        mean_dice = np.mean(dice_scores) if dice_scores else 0.0
        print(f"  Validation Metrics → mIoU: {mean_iou:.4f}, mDice: {mean_dice:.4f}")
        return mean_iou, mean_dice


class maskrcnn_final(MaskRCNNModel):
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.cfg = json.load(f)
        # Extract parameters
        self.num_classes = self.cfg["num_classes"]
        self.data_path = self.cfg["input_data_path"]
        self.batch_size = self.cfg.get("batch_size", 2)
        self.val_split = self.cfg.get("val_split", 1.0)
        self.learning_rate = self.cfg.get("learning_rate", 0.001)
        self.num_epochs = self.cfg.get("num_epochs", 20)
        self.score_thresh = self.cfg.get("score_threshold", 0.5)
        self.inference_model_path = self.cfg["inference_model_path"]
        self.saved_model_path = self.cfg['saved_model_path']
        self.inference_img = self.cfg.get("inference_img", None)
        self.infer_folder = self.cfg.get("infer_folder", None)
        self.inference_outputfolder = self.cfg.get("inference_outputfolder", "inference_results")
        # Build dataset and dataloaders
        self.train_dataset, self.val_dataset, self.train_loader, self.val_loader, _ = build_dataset_dataloader(
            data_path=self.data_path,
            batch_size=self.batch_size,
            val_split=self.val_split
        )

        # Initialize parent
        super().__init__(num_classes=self.num_classes)
        
        
    def train(self):
        return super().train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            learning_rate=self.learning_rate,
            saved_dir=self.saved_model_path,
            num_epochs=self.num_epochs
        )

    def infer(self):
        if not self.inference_img:
            raise ValueError("No inference_img in config file")
        img_path = self.inference_img
        out_path = os.path.join(self.inference_outputfolder, os.path.splitext(os.path.basename(img_path))[0])
        return super().infer_mask(
            model_path=self.inference_model_path,
            input_path=img_path,
            output_path=out_path,
            score_thresh=self.score_thresh,
            num_classes=self.num_classes
        )
    def infer_batch(self):
        if not self.infer_folder:
            raise ValueError("No infer_folder in config file")
        input_dir = self.infer_folder
        return super().infer_mask_batch(
            model_path=self.inference_model_path,
            input_dir=input_dir,
            output_dir=self.inference_outputfolder,
            score_thresh=self.score_thresh,
            num_classes=self.num_classes
        )


if __name__ == "__main__":

    model = maskrcnn_final(config_file='maskrcnn_config.json')
    #model.train()
    #model.infer()
    model.infer_batch()
    exit()
    # dataset and dataloader
    data_path = '/home/cli74/Desktop/cbi/public/Biohackathon_2025/Datasets/mask_rcnn_new/'
    batch_size = 8
    train_dataset, val_dataset, train_loader, val_loader, num_classes = build_dataset_dataloader(
        data_path = data_path, 
        batch_size  =batch_size, 
        val_split=0.8)

    print('traindataset size:', len(train_dataset))
    if val_dataset is not None:
        print('valdataset size:', len(val_dataset))
    print("Number of classes:", num_classes)
    img, target = train_dataset[0]
    # print(img.shape, img.dtype)
    # print(target["boxes"].shape, target["labels"], target["masks"].shape)
    # print("Image range:", img.min().item(), img.max().item())
    #print("Boxes:", target["boxes"])
    print("Labels:", target["labels"])
    # print("Mask values:", target["masks"].unique())
    #exit()
    # build model
    model = MaskRCNNModel(num_classes=num_classes)
    print(f"Initialized MaskRCNNModel with {num_classes} classes")

    # training
    #model.train(train_loader = train_loader, val_loader = val_loader, learning_rate=0.001, num_epochs=100)
    
    # inference
    img_path = os.path.join(data_path, 'inference/0006.tif')
    model_path = 'checkpoints/maskrcnn_20250930_114823/maskrcnn_best_epoch53.pth'
    #output_path = model.infer_mask(model_path=model_path, input_path=img_path, output_path="predictions", score_thresh=0.6, num_classes=num_classes)
    
    input_folder = os.path.join(data_path, 'inference')
    model.infer_mask_batch(model_path=model_path, input_dir=input_folder, output_dir="predictions", score_thresh=0.6, num_classes=num_classes)
    #output_path = model.save_inference(model_path=model_path, image_path=img_path, score_thresh=0.6)
    
    








    

