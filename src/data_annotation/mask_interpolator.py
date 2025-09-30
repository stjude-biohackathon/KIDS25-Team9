import numpy as np
from scipy import ndimage
import tifffile as tiff
import napari


def morphological_contour_interpolation(volume, annotated_indices, label):
    interpolated_volume = np.zeros_like(volume, dtype=np.uint8)

    for z in annotated_indices:
        mask = (volume[z] == label).astype(np.uint8)
        interpolated_volume[z][mask == 1] = label

    for i in range(len(annotated_indices) - 1):
        z1, z2 = annotated_indices[i], annotated_indices[i + 1]
        mask1 = (volume[z1] == label).astype(np.uint8)
        mask2 = (volume[z2] == label).astype(np.uint8)

        for z in range(z1 + 1, z2):
            alpha = (z - z1) / (z2 - z1)
            dist1 = ndimage.distance_transform_edt(mask1)
            dist2 = ndimage.distance_transform_edt(mask2)
            interp_dist = (1 - alpha) * dist1 + alpha * dist2
            interp_mask = (interp_dist > 0).astype(np.uint8)
            interpolated_volume[z][interp_mask == 1] = label

    return interpolated_volume


def sdf_interpolation(volume, annotated_indices, label):
    interpolated_volume = np.zeros_like(volume, dtype=np.uint8)

    for z in annotated_indices:
        mask = (volume[z] == label).astype(np.uint8)
        interpolated_volume[z][mask == 1] = label

    for i in range(len(annotated_indices) - 1):
        z1, z2 = annotated_indices[i], annotated_indices[i + 1]
        mask1 = (volume[z1] == label).astype(np.uint8)
        mask2 = (volume[z2] == label).astype(np.uint8)

        sdf1 = ndimage.distance_transform_edt(mask1) - ndimage.distance_transform_edt(1 - mask1)
        sdf2 = ndimage.distance_transform_edt(mask2) - ndimage.distance_transform_edt(1 - mask2)

        for z in range(z1 + 1, z2):
            alpha = (z - z1) / (z2 - z1)
            sdf_interp = (1 - alpha) * sdf1 + alpha * sdf2
            interp_mask = (sdf_interp > 0).astype(np.uint8)
            interpolated_volume[z][interp_mask == 1] = label

    return interpolated_volume


def load_tiff(path):
    return tiff.imread(path)


def save_tiff(path, volume):
    tiff.imwrite(path, volume)


def display_napari(*volumes, names=None):
    viewer = napari.Viewer()
    if names is None:
        names = [f'Volume {i + 1}' for i in range(len(volumes))]
    for vol, name in zip(volumes, names):
        viewer.add_labels(vol, name=name)
    napari.run()

def find_indices_by_label(volume, label):
    """
    Find sorted list of z-indices where the given label is present in the 3D volume.
    Args:
        volume (np.ndarray): 3D numpy array (z, y, x).
        label (int): Label value to search for.
    Returns:
        List[int]: Sorted list of z-indices where label is present.
    """
    indices = []
    for z in range(volume.shape[0]):
        if np.any(volume[z] == label):
            indices.append(z)
    return sorted(indices)

def interpolate_mask(volume, label):
    annotated_indices = find_indices_by_label(volume, label)
    return sdf_interpolation(volume, annotated_indices, label)

if __name__ == "__main__":
    # Path to input TIFF
    input_tif = "/Users/nshakya/2_labels.tif"
    output_morph_tif = "interpolated_morph.tif"
    output_sdf_tif = "interpolated_sdf.tif"

    # Load 3D TIFF
    volume = load_tiff(input_tif)

    # Annotated slices (example: user-provided)
    label = 2
    annotated_indices = find_indices_by_label(volume, label)

    # Run interpolations
    interpolated_morph = morphological_contour_interpolation(volume, annotated_indices, label)
    interpolated_sdf = sdf_interpolation(volume, annotated_indices, label)

    # Save outputs as 3D TIFF
    # save_tiff(output_morph_tif, interpolated_morph)
    save_tiff(output_sdf_tif, interpolated_sdf)

    print(f"Morphological interpolation saved to {output_morph_tif}")
    print(f"SDF interpolation saved to {output_sdf_tif}")

    # # Display in Napari
    display_napari(interpolated_morph, interpolated_sdf,
                   names=['Morphological Interpolation', 'SDF Interpolation'])
