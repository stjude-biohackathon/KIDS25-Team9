import numpy as np
import os
import uuid
from glob import glob

import albumentations as A
import cv2
import tifffile
import random


class ImageAugmentor:
    def __init__(self, input_img_dir, input_lbl_dir, output_img_dir, output_lbl_dir, transform_types=None,
                 iterations=100, params=None):
        """
        Args:
            input_img_dir (str): Path to input images folder
            input_lbl_dir (str): Path to input labels or labels_instance folder
            output_img_dir (str): Path to output images folder
            output_lbl_dir (str): Path to output labels or labels_instance folder
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
        self.image_dir = input_img_dir
        self.output_image_dir = output_img_dir
        self.output_label_dir = output_lbl_dir
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)

        # Determine label mode based on input_lbl_dir contents
        # If input_lbl_dir contains only .tif files -> "labels"
        # If input_lbl_dir contains subfolders (and those subfolders contain .tif files) -> "labels_instance"
        tif_files = [f for f in os.listdir(input_lbl_dir) if f.lower().endswith('.tif')]
        subdirs = [f for f in os.listdir(input_lbl_dir) if os.path.isdir(os.path.join(input_lbl_dir, f))]
        if tif_files and not subdirs:
            self.label_dir = input_lbl_dir
            self.label_mode = "labels"
        elif subdirs:
            self.label_instance_dir = input_lbl_dir
            self.label_mode = "labels_instance"
        else:
            raise ValueError(
                "No valid label directory found in input_lbl_dir. Expected .tif files or subfolders containing .tif files.")

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
        max_pixel_value = normalize_params.get("max_pixel_value", None)

        if max_pixel_value is None:
            # Automatically determine max_pixel_value from first image in self.image_dir
            image_files = sorted(glob(os.path.join(self.image_dir, "*.tif")))
            if not image_files:
                max_pixel_value = 255.0
            else:
                first_img = tifffile.imread(image_files[0])
                dtype = first_img.dtype
                if np.issubdtype(dtype, np.uint8):
                    max_pixel_value = 2 ** 8 - 1
                elif np.issubdtype(dtype, np.uint16):
                    max_pixel_value = 2 ** 16 - 1
                elif np.issubdtype(dtype, np.uint32):
                    max_pixel_value = 2 ** 32 - 1
                elif np.issubdtype(dtype, np.float32):
                    max_pixel_value = 1.0
                else:
                    max_pixel_value = 255.0

        # Ensure Normalize only affects the image; masks will be ignored by Albumentations
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
        if self.label_mode == "labels":
            image_files = sorted(glob(os.path.join(self.image_dir, "*.tif")))
            label_files = sorted(glob(os.path.join(self.label_dir, "*.tif")))

            for img_path, lbl_path in zip(image_files, label_files):
                for i in range(self.iterations):
                    self._augment_pair(img_path, lbl_path, i + 1)

            print(f"✅ Augmentation complete! Results saved to {os.path.dirname(self.output_image_dir)}")

        elif self.label_mode == "labels_instance":
            # Get all subfolders in Labels_instance
            subfolders = [f.path for f in os.scandir(self.label_instance_dir) if f.is_dir()]
            subfolder_names = [os.path.basename(f) for f in subfolders]

            # For each image in Images directory
            image_files = sorted(glob(os.path.join(self.image_dir, "*.tif")))
            for img_path in image_files:
                basename = os.path.splitext(os.path.basename(img_path))[0]

                # Collect all matching label paths from subfolders
                label_paths = []
                for subfolder, subfolder_name in zip(subfolders, subfolder_names):
                    label_path = os.path.join(subfolder, f"{basename}.tif")
                    if os.path.exists(label_path):
                        label_paths.append((label_path, subfolder_name))

                if not label_paths:
                    continue  # No labels found for this image, skip

                for i in range(self.iterations):
                    # Generate transform parameters once per iteration and build transform
                    transform, params = self._generate_transform_params()
                    # Apply augmentation to image and all labels with the same transform
                    self._augment_pair(img_path, label_paths, i + 1, transform, params)

            print(f"✅ Augmentation complete! Results saved to {os.path.dirname(self.output_image_dir)}")

    def _generate_transform_params(self):
        types = set(self.transform_types)

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

        transform = self._build_transform(
            rotate=rotate,
            scale=scale,
            translate_percent=translate_percent,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma,
            erasing_scale=erasing_scale,
            erasing_ratio=erasing_ratio
        )
        params = {
            "rotate": rotate,
            "scale": scale,
            "translate_percent": translate_percent,
            "elastic_alpha": elastic_alpha,
            "elastic_sigma": elastic_sigma,
            "erasing_scale": erasing_scale,
            "erasing_ratio": erasing_ratio,
        }
        return transform, params

    def _augment_pair(self, img_path, label_paths, iteration, transform=None, params=None):
        """
        Wrapper method to augment an image and one or multiple labels with the same transform.
        label_paths: either a single label path string or a list of tuples (label_path, subfolder_name)
        """
        if transform is None or params is None:
            # If called without transform, generate one and apply to single pair
            types = set(self.transform_types)

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
                translate_percent = random.uniform(trans_range[0], trans_range[1])
            else:
                translate_percent = 0

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

            transform = self._build_transform(
                rotate=rotate,
                scale=scale,
                translate_percent=translate_percent,
                elastic_alpha=elastic_alpha,
                elastic_sigma=elastic_sigma,
                erasing_scale=erasing_scale,
                erasing_ratio=erasing_ratio
            )
            params = {
                "rotate": rotate,
                "scale": scale,
                "translate_percent": translate_percent,
                "elastic_alpha": elastic_alpha,
                "elastic_sigma": elastic_sigma,
                "erasing_scale": erasing_scale,
                "erasing_ratio": erasing_ratio,
            }

        # If label_paths is a single string, convert to list of one tuple for uniform processing
        if isinstance(label_paths, str):
            label_paths = [(label_paths, None)]

        # Generate a unique_id once per iteration
        unique_id = uuid.uuid4().hex[:8]

        # Augment image once
        self._augment_with_transform(img_path, iteration, transform, params, is_image=True, unique_id=unique_id)

        # Augment each label with same transform
        for lbl_path, subfolder_name in label_paths:
            # Determine output label directory
            if subfolder_name is not None:
                label_out_dir = os.path.join(self.output_label_dir, subfolder_name)
                os.makedirs(label_out_dir, exist_ok=True)
            else:
                label_out_dir = self.output_label_dir
            self._augment_with_transform(lbl_path, iteration, transform, params, is_image=False,
                                         label_out_dir=label_out_dir, unique_id=unique_id)

    def _augment_with_transform(self, path, iteration, transform, params, is_image=True, label_out_dir=None,
                                unique_id=None, return_array=False):
        """Apply the provided transform to a single image or label file.
        For 3D images, applies the transform slice by slice, handling channels appropriately.
        If return_array is True, return the augmented array instead of writing to file.
        """
        if label_out_dir is None:
            label_out_dir = self.output_label_dir

        stack = tifffile.imread(path)

        # Detect if images are 2D (single image)
        is_2d = stack.ndim == 2

        aug_slices = []
        if is_2d:
            # Convert 2D grayscale image to (H, W, 1) for albumentations
            slice_ = stack
            # Ensure dtype is float32 or uint8 for albumentations
            if slice_.dtype not in [np.float32, np.uint8]:
                if is_image:
                    slice_ = slice_.astype(np.float32)
                else:
                    slice_ = slice_.astype(np.uint8)
            slice_exp = slice_[..., None]
            if is_image:
                augmented = transform(image=slice_exp)
                aug = augmented["image"]
            else:
                augmented = transform(image=slice_exp, mask=slice_exp)
                aug = augmented["mask"]
            # Remove channel dimension for 2D input
            aug = aug[..., 0]
            aug_slices.append(aug)
        else:
            # For 3D images, process each slice (axis 0) individually
            for i in range(stack.shape[0]):
                slice_ = stack[i]
                # Ensure slice is 2D
                if slice_.ndim != 2:
                    raise ValueError(f"Expected 2D slice at index {i}, got shape {slice_.shape}")
                # Ensure dtype is float32 or uint8 for albumentations
                if slice_.dtype not in [np.float32, np.uint8]:
                    if is_image:
                        slice_ = slice_.astype(np.float32)
                    else:
                        slice_ = slice_.astype(np.uint8)
                # Add channel dimension
                slice_exp = slice_[..., None]
                if is_image:
                    augmented = transform(image=slice_exp)
                    aug = augmented["image"]
                else:
                    augmented = transform(image=slice_exp, mask=slice_exp)
                    aug = augmented["mask"]
                # Remove extra channel dimension
                aug = aug[..., 0]
                aug_slices.append(aug)
        # Stack slices back for 3D images
        if is_2d:
            result = aug_slices[0]
        else:
            result = np.stack(aug_slices, axis=0)

        if return_array:
            return result
        else:
            basename = os.path.splitext(os.path.basename(path))[0]
            if is_image:
                out_path = os.path.join(self.output_image_dir, f"{basename}_{iteration}_{unique_id}.tif")
            else:
                out_path = os.path.join(label_out_dir, f"{basename}_{iteration}_{unique_id}.tif")
            tifffile.imwrite(out_path, result)

    def transform_sample(self, n: int=0) -> list:
        """
        Apply a random transform (same as for augment) to the nth image and corresponding label(s), returning arrays.
        For 'labels', picks nth image and nth label.
        For 'labels_instance', picks nth image and all corresponding labels in subfolders.
        Returns: [image_array, [label_array1, label_array2, ...]]
        """
        if self.label_mode == "labels":
            image_files = sorted(glob(os.path.join(self.image_dir, "*.tif")))
            label_files = sorted(glob(os.path.join(self.label_dir, "*.tif")))
            if n < 0 or n >= len(image_files) or n >= len(label_files):
                raise IndexError(f"Index n={n} is out of range for available images/labels.")
            img_path = image_files[n]
            lbl_path = label_files[n]
            transform, params = self._generate_transform_params()
            img_array = self._augment_with_transform(img_path, 1, transform, params, is_image=True, return_array=True)
            lbl_array = self._augment_with_transform(lbl_path, 1, transform, params, is_image=False, return_array=True)
            return [img_array, [lbl_array]]
        elif self.label_mode == "labels_instance":
            # Get all subfolders in Labels_instance
            subfolders = [f.path for f in os.scandir(self.label_instance_dir) if f.is_dir()]
            subfolder_names = [os.path.basename(f) for f in subfolders]
            image_files = sorted(glob(os.path.join(self.image_dir, "*.tif")))
            if n < 0 or n >= len(image_files):
                raise IndexError(f"Index n={n} is out of range for available images.")
            img_path = image_files[n]
            basename = os.path.splitext(os.path.basename(img_path))[0]
            # Collect all matching label paths from subfolders
            label_paths = []
            for subfolder in subfolders:
                label_path = os.path.join(subfolder, f"{basename}.tif")
                if os.path.exists(label_path):
                    label_paths.append(label_path)
            if not label_paths:
                raise FileNotFoundError(f"No labels found for image {img_path} in any subfolder.")
            transform, params = self._generate_transform_params()
            img_array = self._augment_with_transform(img_path, 1, transform, params, is_image=True, return_array=True)
            lbl_arrays = []
            for lbl_path in label_paths:
                lbl_array = self._augment_with_transform(lbl_path, 1, transform, params, is_image=False,
                                                         return_array=True)
                lbl_arrays.append(lbl_array)
            return [img_array, lbl_arrays]
        else:
            raise ValueError("Unknown label mode.")


def view_in_napari(img_array, lbl_arrays):
    import napari
    # Open napari viewer
    viewer = napari.Viewer()
    # Add the image
    viewer.add_image(img_array, name='Image')
    # Add labels
    for i, lbl in enumerate(lbl_arrays):
        viewer.add_labels(lbl, name=f'Label_{i}')
    napari.run()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    test_cases = [
        ("multisegment", "Labels_instance"),
        # ("3d", "Labels"),
        # ("2d", "Labels")
    ]
    for data_type, label in test_cases:
        augmentor = ImageAugmentor(
            f'dataset/{data_type}/Images',
            f'dataset/{data_type}/{label}',
            f'augmented/{data_type}/Images',
            f'augmented/{data_type}/{label}',
            transform_types=[
                # "rotation",
                # "translation",
                #  "elastic",
                #  "erasing",
                # "scaling",
                # "normalize",
                # "horizontal_flip",
                "vertical_flip"
            ],
            iterations=2,
            params={
                "rotation": {"degrees": (-60, 60)},
                "translation": {"shift": (-0.1, 0.1)},
                "elastic": {"alpha": 1.0, "sigma": 50.0},
                "erasing": {"scale": (0.02, 0.33), "ratio": (0.3, 3.3), "p": 1.0},
                "normalize": {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0), "max_pixel_value": 255.0, "p": 1.0}
            }
        )
        # augmentor.augment_dataset()

        img_array, lbl_arrays = augmentor.transform_sample()
        view_in_napari(img_array, lbl_arrays)