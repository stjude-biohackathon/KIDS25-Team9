import numpy as np
from scipy import ndimage
import tifffile as tiff
import napari


# add these imports near the top of your file
from skimage import measure
from skimage.draw import polygon
from scipy.interpolate import interp1d

# --- helpers ---------------------------------------------------------------

def resample_contour(contour, n_points=200):
    """
    Resample a (N,2) contour (row, col) to n_points equally spaced along arc length.
    """
    contour = np.asarray(contour, dtype=float)
    if contour.shape[0] < 2:
        return np.tile(contour[0], (n_points, 1))

    # make closed if not already
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))

    deltas = np.diff(contour, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cum = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total = cum[-1]
    if total <= 0:
        return np.tile(contour[0], (n_points, 1))

    # sample equally spaced along length
    tgt = np.linspace(0.0, total, n_points, endpoint=False)
    xs = contour[:, 1]   # cols
    ys = contour[:, 0]   # rows
    fx = interp1d(cum, xs, kind='linear')
    fy = interp1d(cum, ys, kind='linear')
    xs_s = fx(tgt)
    ys_s = fy(tgt)
    pts = np.vstack((ys_s, xs_s)).T  # (row, col)
    return pts


def _components_with_contours(slice_mask):
    """
    Given a 2D binary mask, return a list of components with 'mask', 'contour', 'centroid'.
    contour coordinates are in image coordinates (row, col).
    """
    comps = []
    labeled = measure.label(slice_mask)
    for region in measure.regionprops(labeled):
        comp_mask = (labeled == region.label).astype(np.uint8)
        contours = measure.find_contours(comp_mask, 0.5)
        if not contours:
            continue
        # choose the longest contour (most likely the external boundary)
        lengths = [np.sum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))) for c in contours]
        contour = contours[int(np.argmax(lengths))]
        comps.append({'mask': comp_mask, 'contour': contour, 'centroid': region.centroid})
    return comps

# --- main morphological contour interpolation ------------------------------

def morphological_contour_interpolation(volume, annotated_indices, label, n_points=200):
    """
    Interpolate masks between annotated z-slices by interpolating object contours.
    - volume: 3D labeled volume (z, y, x)
    - annotated_indices: sorted list of z indices that are annotated for `label`
    - label: integer label value to interpolate
    - n_points: number of contour samples per object (controls smoothness)
    Returns a 3D volume containing only `label` (all other voxels = 0).
    """
    if len(annotated_indices) == 0:
        return np.zeros_like(volume, dtype=np.uint8)

    Z, H, W = volume.shape
    interpolated = np.zeros_like(volume, dtype=np.uint8)

    annotated_indices = sorted(annotated_indices)

    # copy exact annotated slices
    for z in annotated_indices:
        interpolated[z][volume[z] == label] = label

    # process each annotated interval
    for idx in range(len(annotated_indices) - 1):
        z1 = annotated_indices[idx]
        z2 = annotated_indices[idx + 1]
        mask1 = (volume[z1] == label).astype(np.uint8)
        mask2 = (volume[z2] == label).astype(np.uint8)

        comps1 = _components_with_contours(mask1)
        comps2 = _components_with_contours(mask2)

        # greedy match components by centroid (simple & fast)
        available2 = list(range(len(comps2)))
        matches = []
        for i in range(len(comps1)):
            if not available2:
                matches.append((i, None))
                continue
            # compute distances to remaining comps2
            dists = [np.linalg.norm(np.array(comps1[i]['centroid']) - np.array(comps2[j]['centroid']))
                     for j in available2]
            j_selected = available2[int(np.argmin(dists))]
            matches.append((i, j_selected))
            available2.remove(j_selected)
        # any leftover comps2 that weren't matched
        for j in available2:
            matches.append((None, j))

        # now interpolate each matched pair
        for (i, j) in matches:
            if i is not None:
                contour1 = comps1[i]['contour']
                pts1 = resample_contour(contour1, n_points=n_points)
                centroid1 = np.array(comps1[i]['centroid'], dtype=float)
            else:
                pts1 = None
                centroid1 = None

            if j is not None:
                contour2 = comps2[j]['contour']
                pts2 = resample_contour(contour2, n_points=n_points)
                centroid2 = np.array(comps2[j]['centroid'], dtype=float)
            else:
                pts2 = None
                centroid2 = None

            # both missing -> nothing to do
            if pts1 is None and pts2 is None:
                continue

            # if one side is missing, collapse to centroid of the available component
            if pts1 is None:
                pts1 = np.tile(centroid2, (n_points, 1))
            if pts2 is None:
                pts2 = np.tile(centroid1, (n_points, 1))

            # for every intermediate z, interpolate points and rasterize polygon
            for z in range(z1 + 1, z2):
                alpha = (z - z1) / float(z2 - z1)
                pts = (1.0 - alpha) * pts1 + alpha * pts2  # shape (n_points, 2) rows, cols

                # clamp polygon coordinates inside image (polygon tolerates floats)
                pts[:, 0] = np.clip(pts[:, 0], 0, H - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, W - 1)

                rr, cc = polygon(pts[:, 0], pts[:, 1], shape=(H, W))
                interpolated[z][rr, cc] = label

    return interpolated


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

def interpolate_mask(volume, label, interpolation_type="sdf"):
    annotated_indices = find_indices_by_label(volume, label)
    if interpolation_type == "sdf":
        return sdf_interpolation(volume, annotated_indices, label)
    else:
        return morphological_contour_interpolation(volume, annotated_indices, label)


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
