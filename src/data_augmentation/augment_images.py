import os
from glob import glob

import albumentations as A
import cv2
import tifffile
import random


class ImageAugmentor:
    def __init__(self, input_dir, output_dir, transform_types=None, iterations=100, params=None):
        """
        Args:
            input_dir (str): Path to input folder
            output_dir (str): Path to output base folder
            transform_types (list of str): List containing one or more of ["rotation", "scaling", "translation", "elastic", "horizontal_flip", "vertical_flip", "erasing", "normalize", "all"]
            iterations (int): Number of augmented versions per input image
            params (dict): Parameters for augmentation ranges. Examples:
                {
                    "rotation": {"degrees": (-45, 45)},
                    "scaling": {"scale": (0.8, 1.2)},
                    "translation": {"shift": (-0.1, 0.1)},
                    "elastic": {"alpha": 1.0, "sigma": 50.0, "alpha_affine": 10.0},
                    "erasing": {"scale": (0.02, 0.33), "ratio": (0.3, 3.3), "p": 1.0},
                    "normalize": {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0), "max_pixel_value": 255.0, "p": 1.0}
                }
        """
        self.image_dir = os.path.join(input_dir, "Images")
        self.label_dir = os.path.join(input_dir, "Images")
        self.output_image_dir = os.path.join(output_dir, "Images")
        self.output_label_dir = os.path.join(output_dir, "Labels")
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)

        if transform_types is None:
            transform_types = ["all"]
        valid_types = {"rotation", "scaling", "translation", "elastic", "horizontal_flip", "vertical_flip", "erasing",
                       "normalize", "all"}
        for t in transform_types:
            if t not in valid_types:
                raise ValueError(
                    f"Invalid transform_type '{t}'. Choose from: rotation, scaling, translation, elastic, horizontal_flip, vertical_flip, erasing, normalize, all")
        self.transform_types = transform_types
        self.iterations = iterations
        self.params = params if params else {}

    def _build_affine_transform(self, rotate, scale, translate_percent):
        return A.Affine(
            rotate=rotate,
            scale=scale,
            translate_percent=translate_percent,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,  # background fill for images
            fill_mask=0,  # background fill for masks
            p=1.0,
            interpolation=cv2.INTER_CUBIC,
            mask_interpolation=cv2.INTER_NEAREST
        )

    def _build_flip_transforms(self):
        transforms = []
        types = set(self.transform_types)
        if "horizontal_flip" in types or "all" in types:
            transforms.append(A.HorizontalFlip(p=1.0))
        if "vertical_flip" in types or "all" in types:
            transforms.append(A.VerticalFlip(p=1.0))
        return transforms

    def _build_elastic_transform(self, alpha, sigma):
        return A.ElasticTransform(alpha=alpha, sigma=sigma, interpolation=cv2.INTER_CUBIC,
                                  mask_interpolation=cv2.INTER_NEAREST,
                                  border_mode=cv2.BORDER_CONSTANT, fill=0,
                                  fill_mask=0, p=1.0)

    def _build_erasing_transform(self, scale, ratio):
        return A.Erasing(
            scale=scale,
            ratio=ratio,
            p=1.0
        )

    def _build_normalize_transform(self):
        normalize_params = self.params.get("normalize", {})
        mean = normalize_params.get("mean", (0.0, 0.0, 0.0))
        std = normalize_params.get("std", (1.0, 1.0, 1.0))
        max_pixel_value = normalize_params.get("max_pixel_value", 255.0)
        return A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0)

    def _build_transform(self, rotate=0, scale=1.0, translate_percent=(0, 0), elastic_alpha=None, elastic_sigma=None,
                         erasing_scale=None, erasing_ratio=None):
        """Build a composed transform based on all types in self.transform_types, applied sequentially, using fixed parameters."""
        types = set(self.transform_types)
        transforms = []

        if ("rotation" in types or "all" in types) or ("scaling" in types or "all" in types) or (
                "translation" in types or "all" in types):
            affine_transform = self._build_affine_transform(rotate, scale, translate_percent)
            transforms.append(affine_transform)

        transforms.extend(self._build_flip_transforms())

        if "elastic" in types or "all" in types:
            if elastic_alpha is None:
                elastic_alpha = self.params.get("elastic", {}).get("alpha", 1.0)
            if elastic_sigma is None:
                elastic_sigma = self.params.get("elastic", {}).get("sigma", 50.0)
            elastic_transform = self._build_elastic_transform(elastic_alpha, elastic_sigma)
            transforms.append(elastic_transform)

        if "erasing" in types or "all" in types:
            if erasing_scale is None:
                erasing_scale = self.params.get("erasing", {}).get("scale", (0.02, 0.33))
            if erasing_ratio is None:
                erasing_ratio = self.params.get("erasing", {}).get("ratio", (0.3, 3.3))
            transforms.append(self._build_erasing_transform(erasing_scale, erasing_ratio))

        if "normalize" in types or "all" in types:
            transforms.append(self._build_normalize_transform())

        return A.Compose(transforms)

    def augment_dataset(self):
        """Apply augmentation to all images/labels with multiple iterations"""
        image_files = sorted(glob(os.path.join(self.image_dir, "*.tif")))
        label_files = sorted(glob(os.path.join(self.label_dir, "*.tif")))

        for img_path, lbl_path in zip(image_files, label_files):
            for i in range(self.iterations):
                self._augment_pair(img_path, lbl_path, i + 1)

        print(f"✅ Augmentation complete! Results saved to {os.path.dirname(self.output_image_dir)}")

    def _augment_pair(self, img_path, lbl_path, iteration):
        """Apply one augmentation to a single image–label pair"""
        image_stack = tifffile.imread(img_path)
        label_stack = tifffile.imread(lbl_path)

        # Detect if images are 2D (single image)
        is_2d = image_stack.ndim == 2

        if is_2d:
            # Convert 2D grayscale image to (H, W, 1) for albumentations
            image_slices = [image_stack[..., None]]
            label_slices = [label_stack[..., None] if label_stack.ndim == 2 else label_stack]
        else:
            image_slices = image_stack
            label_slices = label_stack

        types = set(self.transform_types)

        # Sample fixed values once for this iteration
        if "rotation" in types or "all" in types:
            rot_range = self.params.get("rotation", {}).get("degrees", (-30, 30))
            rotate = random.uniform(rot_range[0], rot_range[1])
        else:
            rotate = 0

        if "scaling" in types or "all" in types:
            scale_range = self.params.get("scaling", {}).get("scale", (0.9, 1.1))
            scale = random.uniform(scale_range[0], scale_range[1])
        else:
            scale = 1.0

        if "translation" in types or "all" in types:
            trans_range = self.params.get("translation", {}).get("shift", (-0.05, 0.05))
            translate_percent = (
            random.uniform(trans_range[0], trans_range[1]), random.uniform(trans_range[0], trans_range[1]))
        else:
            translate_percent = (0, 0)

        if "elastic" in types or "all" in types:
            elastic_params = self.params.get("elastic", {})
            elastic_alpha = elastic_params.get("alpha", 1.0)
            elastic_sigma = elastic_params.get("sigma", 50.0)
        else:
            elastic_alpha = None
            elastic_sigma = None

        if "erasing" in types or "all" in types:
            erasing_params = self.params.get("erasing", {})
            erasing_scale = erasing_params.get("scale", (0.02, 0.33))
            erasing_ratio = erasing_params.get("ratio", (0.3, 3.3))
        else:
            erasing_scale = None
            erasing_ratio = None

        # Build a single transform for this entire stack (per iteration) with fixed parameters
        transform = self._build_transform(
            rotate=rotate,
            scale=scale,
            translate_percent=translate_percent,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma,
            erasing_scale=erasing_scale,
            erasing_ratio=erasing_ratio
        )

        aug_image_slices = []
        aug_label_slices = []

        # Apply the same transform to all slices in the stack
        for img_slice, lbl_slice in zip(image_slices, label_slices):
            augmented = transform(image=img_slice, mask=lbl_slice)
            aug_img = augmented["image"]
            aug_lbl = augmented["mask"]

            # Remove channel dimension if 2D input
            if is_2d:
                aug_img = aug_img[..., 0]
                if aug_lbl.ndim == 3 and aug_lbl.shape[-1] == 1:
                    aug_lbl = aug_lbl[..., 0]

            aug_image_slices.append(aug_img)
            aug_label_slices.append(aug_lbl)

        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_img_path = os.path.join(self.output_image_dir, f"{basename}_{iteration}.tif")
        out_lbl_path = os.path.join(self.output_label_dir, f"{basename}_{iteration}.tif")

        if is_2d:
            tifffile.imwrite(out_img_path, aug_image_slices[0])
            tifffile.imwrite(out_lbl_path, aug_label_slices[0])
        else:
            tifffile.imwrite(out_img_path, aug_image_slices)
            tifffile.imwrite(out_lbl_path, aug_label_slices)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    input_dir = "dataset/3d"
    output_dir = "augmented/3d"

    # Example: apply a combination of rotation, translation, flips, elastic deformation, erasing, and normalization.
    augmentor = ImageAugmentor(
        input_dir, output_dir,
        transform_types=["rotation"],
        iterations=2,
        params={
            "rotation": {"degrees": (-60, 60)},
            "translation": {"shift": (-0.1, 0.1)},
            "elastic": {"alpha": 1.0, "sigma": 50.0},
            "erasing": {"scale": (0.02, 0.33), "ratio": (0.3, 3.3), "p": 1.0},
            "normalize": {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0), "max_pixel_value": 255.0, "p": 1.0}
        }
    )
    augmentor.augment_dataset()
