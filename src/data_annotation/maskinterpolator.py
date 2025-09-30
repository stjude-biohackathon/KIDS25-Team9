import os
import SimpleITK as sitk


class MaskInterpolator3D:
    def __init__(self, input_file: str, file_ext: str = ".tif"):
        self.input_file = input_file
        self.file_ext = file_ext
        self.volume = self._load_mask(input_file)

    def _load_mask(self, path: str):
        """Load a 3D mask using SimpleITK."""
        return sitk.ReadImage(path)

    def _save_mask(self, mask, path: str):
        """Save a mask using SimpleITK."""
        sitk.WriteImage(mask, path)

    def interpolate_slices(self, z_start: int, z_end: int, output_folder: str):
        """
        Interpolates masks between z_start and z_end slices (exclusive).
        Saves intermediate slices in output_folder.
        """
        # Validate z indices
        size = self.volume.GetSize()  # (width, height, depth)
        if z_start < 0 or z_end >= size[2] or z_start >= z_end:
            raise ValueError("Invalid z_start or z_end values.")

        num_slices = z_end - z_start - 1
        if num_slices <= 0:
            print("No slices to interpolate.")
            return

        # Extract start and end slices
        mask_start = sitk.Extract(self.volume, (size[0], size[1], 1), (0, 0, z_start))
        mask_end = sitk.Extract(self.volume, (size[0], size[1], 1), (0, 0, z_end))

        # Reset origin of the extracted slices before joining
        mask_start.SetOrigin((0.0, 0.0))
        mask_end.SetOrigin((0.0, 0.0))

        # Create 3D volume with only start and end slices
        volume_2slices = sitk.JoinSeries([mask_start, mask_end])

        # Resample along z to include intermediate slices
        resampler = sitk.ResampleImageFilter()
        new_size = (size[0], size[1], num_slices + 2)  # +2 includes start and end
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled = resampler.Execute(volume_2slices)

        # Save intermediate slices
        os.makedirs(output_folder, exist_ok=True)
        for i in range(1, new_size[2] - 1):  # skip start and end
            slice_i = sitk.Extract(resampled, (new_size[0], new_size[1], 1), (0, 0, i))
            filename = f"slice_{z_start + i:04d}{self.file_ext}"
            path_new = os.path.join(output_folder, filename)
            self._save_mask(slice_i, path_new)
            print(f"Saved interpolated slice: {filename}")


# Example usage
if __name__ == "__main__":
    input_3d_file = "../data_augmentation/augmented/masks/1.tif"
    output_dir = "../data_augmentation/augmented/masks/interpolated"
    interpolator = MaskInterpolator3D(input_3d_file)
    interpolator.interpolate_slices(z_start=1, z_end=10, output_folder=output_dir)
