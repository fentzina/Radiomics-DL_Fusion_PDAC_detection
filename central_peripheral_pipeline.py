#!/usr/bin/env python3
"""
Automated Central vs. Peripheral Subregion Radiomics Pipeline -- batch mode.

This is GIANADOYME_central_peripheral_ORIGINAL.py (single-case, Colab-manual,
min-max normalized, whole-tumor + subregion CoLIAGe) rebuilt to match
batch_coliage_pipeline_FIXED.py (allilouia.py)'s automatic batch driver, but
scoped down to ONLY the central/peripheral subregion analysis:

  1. AUTOMATIC BATCH PROCESSING -- instead of gdown-ing one hardcoded case
     and cloning panorama_labels by hand, this script discovers and unzips
     every batch_*.zip in --data_dir, matches each case against a shared
     --label_dir, and works through all of them (2240 cases across 4
     batches), the same way allilouia.py does. One case failing does not
     stop the rest of its batch or the other batches.

  2. NO WHOLE-TUMOR CoLIAGe, NO NORMALIZATION -- this pipeline's only goal is
     the central-vs-peripheral heterogeneity radiomics, so the whole-tumor
     (whole guide-mask) 26-channel CoLIAGe pass and the 28-channel tensor it
     used to produce have been removed entirely, along with the min-max
     normalization that used to be applied before saving. CoLIAGe now runs
     ONLY on the two subregions.

  3. CENTRAL vs. PERIPHERAL SUBREGION SPLIT (kept, and is now the whole
     pipeline) -- GIANADOYME's "13b. CENTRAL vs. PERIPHERAL SUBREGION SPLIT
     (heterogeneity analysis)" logic is preserved: the guide mask is split
     into central/peripheral subregions by radial distance from its
     centroid, CoLIAGe is run separately on each subregion (reusing the same
     padded CT crop and Collage class), and each subregion's raw (non-
     normalized) 26-length Haralick mean-feature vector is saved, along with
     their difference (heterogeneity vector) and the two subregion masks.
     If a subregion ends up empty or too small for CoLIAGe, the case is
     logged and marked failed -- there is no whole-tumor fallback output
     anymore, since subregion radiomics is this pipeline's only product.

Per-case preprocessing (orientation, resampling, cropping, HU clipping,
Collage class) is unchanged from allilouia.py / GIANADOYME.

Usage:
    python batch_central_peripheral_pipeline.py \
        --data_dir ./data \
        --label_dir ./panorama_cases \
        --output_dir ./outputs

Expects --data_dir to contain one or more batch_*.zip files (e.g. batch_1.zip,
batch_2.zip, batch_3.zip, batch_4.zip), each containing .nii/.nii.gz CT volumes
(possibly nested in subfolders inside the zip). Every batch is unzipped,
scanned for CT volumes, and processed against the single shared --label_dir,
with output written into a common --output_dir.
"""

import os
import sys
import glob
import math
import logging
import argparse
import zipfile
from itertools import product
from enum import Enum

from tqdm import tqdm

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import scipy.ndimage as ndi
from scipy.ndimage import center_of_mass
import mahotas as mt
from scipy import linalg
from skimage.feature import graycomatrix
from skimage.util.shape import view_as_windows

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocess_pipeline')

# --- CONSTANTS ---
TARGET_SHAPE = (128, 128, 128)   # cubic crop
HU_MIN, HU_MAX = -100, 600       # pancreas window
TUMOR_LABEL = 1
PANCREAS_LABEL = 4
SVD_RADIUS = 3
PAD_Z = 1                        # gradient edge margin, applies to the Z axis only
INTERNAL_FRACTION = 0.5          # central-vs-peripheral split radius fraction

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def get_full_stem(path):
    """
    Returns the filename with .nii/.nii.gz stripped, and NOTHING else stripped.
    This matches the original notebook's case-ID convention exactly, e.g.
    '100815_00001.nii.gz' -> '100815_00001'.

    IMPORTANT: earlier versions of this script also stripped the last
    underscore-separated segment (assuming an nnU-Net-style '..._0000'
    modality suffix). That is UNSAFE as a default: the notebook's own example
    filename '100815_00001.nii.gz' has two underscore-separated parts with no
    modality suffix, and blind stripping would collapse it to '100815',
    silently merging it with any other case sharing that first segment
    (e.g. a different study/series for the same patient). Do not reintroduce
    that behaviour here.
    """
    basename = os.path.basename(path)
    if basename.endswith('.nii.gz'):
        return basename[:-7]
    elif basename.endswith('.nii'):
        return basename[:-4]
    return basename.split('.')[0]


def get_case_id_candidates(path):
    """
    Returns ordered candidate case IDs to try when looking up a matching
    label file, safest first:
      1. The full filename stem (matches the original notebook exactly).
      2. The stem with a trailing modality suffix removed, ONLY if that
         suffix looks like a modality code (purely numeric, e.g. '_0000'),
         which is the nnU-Net convention some datasets use for CT files.
    Candidate 2 is a fallback used only if candidate 1 does not find a match
    in the label lookup (see resolve_label_path) -- it is never used to
    blindly overwrite candidate 1.
    """
    full_stem = get_full_stem(path)
    candidates = [full_stem]

    parts = full_stem.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        stripped = "_".join(parts[:-1])
        if stripped not in candidates:
            candidates.append(stripped)

    return candidates


def resolve_label_path(ct_path, label_lookup):
    """
    Tries each case-ID candidate for ct_path against label_lookup, in order
    of safety (exact full-stem match first). Returns (case_id_used,
    label_path) for the first candidate that matches, or (None, None) if
    none match. Never guesses across ambiguous truncations silently -- if a
    fallback candidate is used, the caller logs it so a mismatch is visible
    in the run log rather than silently corrupting output.
    """
    for candidate in get_case_id_candidates(ct_path):
        if candidate in label_lookup:
            return candidate, label_lookup[candidate]
    return None, None


def preprocess_to_isotropic_lps(sitk_image, is_mask=False, new_spacing=(1.0, 1.0, 1.0)):
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation('LPS')
    sitk_image = orient_filter.Execute(sitk_image)

    orig_spacing = sitk_image.GetSpacing()
    orig_size = sitk_image.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    resample.SetOutputPixelType(sitk.sitkUInt8 if is_mask else sitk.sitkFloat32)
    return resample.Execute(sitk_image)


def get_crop_coords(guide_mask, target_shape):
    if not np.any(guide_mask):
        raise ValueError("Guide mask is empty — cannot compute crop centre.")

    coords = np.argwhere(guide_mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    center_z = (z_min + z_max) // 2
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    orig_z, orig_y, orig_x = guide_mask.shape
    tz, ty, tx = target_shape

    def _clamp(center, size, orig):
        start = max(0, center - size // 2)
        end = start + size
        if end > orig:
            end = orig
            start = max(0, end - size)
        if start < 0:
            start = 0
            end = min(orig, size)
        return start, end

    z_start, z_end = _clamp(center_z, tz, orig_z)
    y_start, y_end = _clamp(center_y, ty, orig_y)
    x_start, x_end = _clamp(center_x, tx, orig_x)
    return z_start, z_end, y_start, y_end, x_start, x_end


# =========================================================================
# COLIAGE CORE CLASS IMPLEMENTATIONS
# =========================================================================

def _svd_dominant_angles(dx, dy, dz, svd_radius):
    is_3D = dx.shape[2] > 1
    svd_diameter = svd_radius * 2 + 1
    window_shape = (svd_diameter, svd_diameter, 1)

    dx_windows = view_as_windows(dx, window_shape)
    dy_windows = view_as_windows(dy, window_shape)
    dz_windows = view_as_windows(dz, window_shape)

    angles_shape = dx_windows.shape[0:3]
    dominant_angles_array = np.zeros(angles_shape + (2 if is_3D else 1,), np.single)

    center_x_range = range(angles_shape[1])
    center_y_range = range(angles_shape[0])
    center_z_range = range(angles_shape[2])
    for x, y, z in product(center_x_range, center_y_range, center_z_range):
        dominant_angles_array[y, x, z, :] = _svd_dominant_angle(x, y, z, dx_windows, dy_windows, dz_windows)

    return dominant_angles_array


def _svd_dominant_angle(x, y, z, dx_windows, dy_windows, dz_windows):
    dx_patch = dx_windows[y, x, z]
    dy_patch = dy_windows[y, x, z]
    dz_patch = dz_windows[y, x, z]

    is_3D = dx_windows.shape[2] > 1
    window_area = dx_patch.size
    flattened_gradients = np.zeros((window_area, (3 if is_3D else 2)))
    matrix_order = 'F'
    flattened_gradients[:, 0] = np.reshape(dx_patch, window_area, order=matrix_order)
    flattened_gradients[:, 1] = np.reshape(dy_patch, window_area, order=matrix_order)
    if is_3D:
        flattened_gradients[:, 2] = np.reshape(dz_patch, window_area, order=matrix_order)

    _, _, v = linalg.svd(flattened_gradients)

    dominant_y = v[0, 0]
    dominant_x = v[1, 0]
    dominant_angle = math.atan2(dominant_y, dominant_x)

    if is_3D:
        dominant_z = v[2, 0]
        secondary_angle = math.atan2(dominant_z, math.sqrt(dominant_x ** 2 + dominant_y ** 2))
        return (dominant_angle, secondary_angle)
    else:
        return dominant_angle


class Collage:
    def __init__(self, img_array, mask_array, svd_radius=5, cooccurence_angles=[x * np.pi/4 for x in range(8)], num_unique_angles=64):
        self._haralick_window_size = svd_radius * 2 + 1
        self._svd_radius = svd_radius
        self._num_unique_angles = num_unique_angles
        self._cooccurence_angles = cooccurence_angles
        self._is_3D = img_array.ndim == 3

        self._img_array = img_array if self._is_3D else img_array.reshape(img_array.shape + (1,))
        thresholded_mask_array = (mask_array != 0).reshape(self._img_array.shape)

        non_zero_indices = np.argwhere(thresholded_mask_array)
        min_mask_coordinates = non_zero_indices.min(0)
        max_mask_coordinates = non_zero_indices.max(0) + 1

        self.mask_min_x, self.mask_min_y, self.mask_min_z = min_mask_coordinates[1], min_mask_coordinates[0], min_mask_coordinates[2]
        self.mask_max_x, self.mask_max_y, self.mask_max_z = max_mask_coordinates[1], max_mask_coordinates[0], max_mask_coordinates[2]

        self._mask_array = thresholded_mask_array[self.mask_min_y:self.mask_max_y, self.mask_min_x:self.mask_max_x, self.mask_min_z:self.mask_max_z]
        self.collage_output = None

    def _calculate_haralick_feature_values(self, img_array, center_x, center_y):
        window_size = self._haralick_window_size
        min_x = int(max(0, center_x - window_size / 2 - 1))
        min_y = int(max(0, center_y - window_size / 2 - 1))
        max_x = int(min(img_array.shape[1] - 1, center_x + window_size / 2 + 1))
        max_y = int(min(img_array.shape[0] - 1, center_y + window_size / 2 + 1))
        cropped_img_array = img_array[min_y:max_y, min_x:max_x]

        cooccurence_matrix = graycomatrix(cropped_img_array, [1], self._cooccurence_angles, levels=self._num_unique_angles)
        cooccurence_matrix = np.sum(cooccurence_matrix, axis=3)[:, :, 0]
        return mt.features.texture.haralick_features([cooccurence_matrix], return_mean=True)

    def _calculate_haralick_textures(self, dominant_angles):
        num_unique_angles = self._num_unique_angles
        dominant_angles_max, dominant_angles_min = dominant_angles.max(), dominant_angles.min()
        dominant_angles_binned = (dominant_angles - dominant_angles_min) / (dominant_angles_max - dominant_angles_min + np.finfo(float).eps) * (num_unique_angles - 1)
        dominant_angles_binned = np.round(dominant_angles_binned).astype(int)

        shape = dominant_angles_binned.shape
        haralick_image = np.empty(shape + (13,))
        haralick_image[:] = np.nan
        height, width, depth = shape

        for z in range(1, depth - 1) if self._is_3D else range(depth):
            for y, x in product(range(height), range(width)):
                if self._mask_array[y, x, z]:
                    haralick_image[y, x, z, :] = self._calculate_haralick_feature_values(dominant_angles_binned[:, :, z], x, y)
        return haralick_image

    def execute(self):
        svd_radius = self._svd_radius
        img_array = self._img_array

        cropped_min_x = max(self.mask_min_x - svd_radius, 0)
        cropped_min_y = max(self.mask_min_y - svd_radius, 0)
        cropped_min_z = max(self.mask_min_z - 1, 0)
        cropped_max_x = min(self.mask_max_x + svd_radius, img_array.shape[1])
        cropped_max_y = min(self.mask_max_y + svd_radius, img_array.shape[0])
        cropped_max_z = min(self.mask_max_z + 1, img_array.shape[2])

        extended_below = self.mask_min_z > 0
        extended_above = self.mask_max_z < img_array.shape[2]

        cropped_image = img_array[cropped_min_y:cropped_max_y, cropped_min_x:cropped_max_x, cropped_min_z:cropped_max_z]

        if cropped_image.max() > 1:
            cropped_image = cropped_image / cropped_image.max()

        dx = np.gradient(cropped_image, axis=1)
        dy = np.gradient(cropped_image, axis=0)
        dz = np.gradient(cropped_image, axis=2) if self._is_3D else np.zeros(dx.shape)

        if extended_below:
            dx, dy, dz = dx[:, :, 1:], dy[:, :, 1:], dz[:, :, 1:]
        if extended_above:
            dx, dy, dz = dx[:, :, :-1], dy[:, :, :-1], dz[:, :, :-1]

        dominant_angles = _svd_dominant_angles(dx, dy, dz, svd_radius)
        angles_shape = dominant_angles.shape

        haralick_features = np.empty(angles_shape[0:3] + (13, 2 if self._is_3D else 1,))
        for angle_index in range(angles_shape[3]):
            haralick_features[:, :, :, :, angle_index] = self._calculate_haralick_textures(dominant_angles[:, :, :, angle_index])

        collage_output = np.empty(img_array.shape + haralick_features.shape[3:5])
        collage_output[:] = np.nan

        collage_output[self.mask_min_y:self.mask_max_y, self.mask_min_x:self.mask_max_x, self.mask_min_z:self.mask_max_z, :, :] = haralick_features

        if not self._is_3D:
            collage_output = np.squeeze(collage_output, 4)
            collage_output = np.squeeze(collage_output, 2)

        self.collage_output = collage_output
        return collage_output


# =========================================================================
# CENTRAL vs. PERIPHERAL HETEROGENEITY HELPERS  (from GIANADOYME "13b")
# =========================================================================

def split_central_peripheral(mask, voxel_spacing=(1.0, 1.0, 1.0), internal_fraction=INTERNAL_FRACTION):
    """
    Splits a binary mask into central and peripheral subregions based on
    radial distance from the mask's center of gravity.

    mask: 3D boolean array, ZYX order
    voxel_spacing: physical spacing per axis, in mm (1,1,1 post-resampling)
    internal_fraction: central region = voxels within internal_fraction * max_radius
                        of the centroid
    Returns: central_mask, peripheral_mask (both boolean, same shape as mask)
    """
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("Mask is empty — no voxels to split.")

    centroid_idx = center_of_mass(mask)  # (z, y, x)

    zz, yy, xx = np.indices(mask.shape)
    dz = (zz - centroid_idx[0]) * voxel_spacing[0]
    dy = (yy - centroid_idx[1]) * voxel_spacing[1]
    dx = (xx - centroid_idx[2]) * voxel_spacing[2]
    dist_from_centroid = np.sqrt(dz**2 + dy**2 + dx**2)

    max_radius = dist_from_centroid[mask].max()
    internal_radius = internal_fraction * max_radius

    central_mask = mask & (dist_from_centroid <= internal_radius)
    peripheral_mask = mask & (dist_from_centroid > internal_radius)

    return central_mask, peripheral_mask


def run_padded_coliage(ct_padded, mask_yxz_bool, svd_radius, pad_z, num_unique_angles=32):
    """
    Runs Collage on ct_padded (already padded (svd_radius, svd_radius, pad_z)
    on each side, YXZ order, shared across whole-tumor and subregion calls)
    against a mask (YXZ, unpadded, same spatial shape as the unpadded crop),
    pads that mask the same way, executes CoLIAGe, and trims the padding
    back out. Returns the (Z, Y, X, 13, 2) Haralick volume.
    """
    mask_padded = np.pad(
        mask_yxz_bool.astype(np.uint8),
        ((svd_radius, svd_radius), (svd_radius, svd_radius), (pad_z, pad_z)),
        mode='constant', constant_values=0,
    )
    instance = Collage(
        ct_padded,
        mask_padded,
        svd_radius=svd_radius,
        num_unique_angles=num_unique_angles,
    )
    instance.execute()

    raw = instance.collage_output
    raw_trimmed = raw[svd_radius:-svd_radius, svd_radius:-svd_radius, pad_z:-pad_z, :, :]

    # Transpose YXZ -> ZYX: (Y, X, Z, 13, 2) -> (Z, Y, X, 13, 2)
    return np.transpose(raw_trimmed, (2, 0, 1, 3, 4))


def summarize_features(haralick_zyx, mask_128):
    """
    Collapses a (128,128,128,13,2) Haralick volume to a single 26-length
    feature vector: the mean of each of the 26 channels, computed only over
    the given subregion mask's own voxels.
    """
    feats = haralick_zyx.reshape(*haralick_zyx.shape[:3], -1)   # (128,128,128,26)
    masked_vals = feats[mask_128]                                # (n_voxels, 26)
    return np.nanmean(masked_vals, axis=0)                       # (26,)


def compute_central_peripheral_heterogeneity(ct_padded, guide_mask_128_zyx, svd_radius, pad_z):
    """
    Runs the "13b" central-vs-peripheral heterogeneity analysis for a single
    case -- this IS the pipeline's output, there is no separate whole-tumor
    pass. Returns a dict with the central vector, peripheral vector,
    heterogeneity vector, and the two subregion masks (all at TARGET_SHAPE
    resolution). Raises if a subregion ends up empty or too small for
    CoLIAGe; callers should let that fail the case, since subregion
    radiomics is the only thing this pipeline produces.
    """
    central_mask, peripheral_mask = split_central_peripheral(
        guide_mask_128_zyx, voxel_spacing=(1.0, 1.0, 1.0), internal_fraction=INTERNAL_FRACTION
    )

    for name, m in [('Central', central_mask), ('Peripheral', peripheral_mask)]:
        if m.sum() == 0:
            raise ValueError(f"{name} subregion mask is empty — cannot run CoLIAGe on it.")
        zs, ys, xs = np.where(m)
        z_extent = zs.max() - zs.min() + 1
        y_extent = ys.max() - ys.min() + 1
        x_extent = xs.max() - xs.min() + 1
        if y_extent < 50 or x_extent < 50 or z_extent < 3:
            raise ValueError(
                f"{name} subregion bbox extent (Z,Y,X)=({z_extent},{y_extent},{x_extent}) "
                f"is below CoLIAGe's minimum size."
            )

    central_mask_yxz = np.transpose(central_mask, (1, 2, 0))
    peripheral_mask_yxz = np.transpose(peripheral_mask, (1, 2, 0))

    central_haralick_zyx = run_padded_coliage(ct_padded, central_mask_yxz, svd_radius, pad_z)
    peripheral_haralick_zyx = run_padded_coliage(ct_padded, peripheral_mask_yxz, svd_radius, pad_z)

    central_vector = summarize_features(central_haralick_zyx, central_mask)
    peripheral_vector = summarize_features(peripheral_haralick_zyx, peripheral_mask)
    heterogeneity_vector = central_vector - peripheral_vector  # simple difference

    return {
        'central_mask': central_mask,
        'peripheral_mask': peripheral_mask,
        'central_vector': central_vector,
        'peripheral_vector': peripheral_vector,
        'heterogeneity_vector': heterogeneity_vector,
    }


# =========================================================================
# CORE PROCESSING PIPELINE
# =========================================================================

def process_case(ct_path, label_path, output_dir, case_id):
    logger.info(f"--- Processing Case: {case_id} ---")

    # 1. Load and resample CT Image
    image = sitk.ReadImage(ct_path)
    image_resampled_sitk = preprocess_to_isotropic_lps(image, is_mask=False)

    # 2. Load Label raw metrics to evaluate classification target
    label_img = nib.load(label_path)
    label_img_data = np.transpose(label_img.get_fdata(), (2, 1, 0))  # XYZ -> ZYX
    unique_labels = np.unique(label_img_data)
    is_pdac_case = (TUMOR_LABEL in unique_labels)

    # 3. Resample Label relative to CT reference grid space
    label_raw = sitk.ReadImage(label_path)
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation('LPS')
    label_lps = orient_filter.Execute(label_raw)

    resampler_labels = sitk.ResampleImageFilter()
    resampler_labels.SetReferenceImage(image_resampled_sitk)
    resampler_labels.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_labels.SetOutputPixelType(sitk.sitkUInt8)
    resampler_labels.SetDefaultPixelValue(0)
    labels_resampled_sitk = resampler_labels.Execute(label_lps)

    # Convert resampled grids to Numpy arrays (ZYX)
    ct_resampled_numpy = sitk.GetArrayFromImage(image_resampled_sitk)
    resampled_labels_numpy = sitk.GetArrayFromImage(labels_resampled_sitk)

    # Build tumor and pancreas segmentation masks
    tumor_3d_mask = (resampled_labels_numpy == TUMOR_LABEL)
    pancreas_3d_mask = (resampled_labels_numpy == PANCREAS_LABEL)
    combined_mask_labels_1_4_3d = tumor_3d_mask | pancreas_3d_mask

    # Define Crop Coordinates
    guide_mask = combined_mask_labels_1_4_3d.astype(bool) if is_pdac_case else pancreas_3d_mask.astype(bool)
    z_start, z_end, y_start, y_end, x_start, x_end = get_crop_coords(guide_mask, TARGET_SHAPE)

    # Clip HU intensities
    clipped_ct_numpy = np.clip(ct_resampled_numpy, HU_MIN, HU_MAX)

    # =========================================================================
    # 1. CROP TO 128^3 FIRST (matching the original notebook exactly)
    # =========================================================================
    if (z_end - z_start, y_end - y_start, x_end - x_start) != TARGET_SHAPE:
        raise ValueError("Crop size did not match TARGET_SHAPE")

    _3d_cropped_ct_zyx   = clipped_ct_numpy[z_start:z_end, y_start:y_end, x_start:x_end]
    _3d_cropped_mask_zyx = guide_mask[z_start:z_end, y_start:y_end, x_start:x_end]

    if _3d_cropped_ct_zyx.shape != _3d_cropped_mask_zyx.shape:
        raise ValueError("CT / mask shape mismatch after crop")
    if not np.any(_3d_cropped_mask_zyx):
        raise ValueError("Cropped mask is empty -- ROI not within crop window.")

    # CoLIAGe expects input in (Height, Width, Depth) -> (Y, X, Z)
    _3d_image_for_collage_yxz = np.transpose(_3d_cropped_ct_zyx, (1, 2, 0))
    _3d_mask_for_collage_yxz  = np.transpose(_3d_cropped_mask_zyx, (1, 2, 0))
    if _3d_image_for_collage_yxz.shape[2] < 3:
        raise ValueError(
            f"Depth {_3d_image_for_collage_yxz.shape[2]} < 3 -- not enough for 3D CoLIAGe"
        )

    # =========================================================================
    # 2. PAD THE 128^3 CROP  (shared CT input for both subregion CoLIAGe runs)
    #    Array axis order here is (Y, X, Z). The SVD dominant-orientation window
    #    is 2D over (Y, X) with diameter 2*SVD_RADIUS+1, so BOTH Y and X need a
    #    full SVD_RADIUS margin to avoid boundary clamping. Z only needs a
    #    1-slice margin (PAD_Z) for the finite-difference gradient at the
    #    volume edge. NOTE: no whole-tumor/whole-guide-mask CoLIAGe pass is
    #    run here -- only the two subregions below are processed.
    # =========================================================================
    ct_padded = np.pad(
        _3d_image_for_collage_yxz,
        ((SVD_RADIUS, SVD_RADIUS), (SVD_RADIUS, SVD_RADIUS), (PAD_Z, PAD_Z)),
        mode='reflect',
    )

    # Create subdirectories and write out matrix files
    img_dir  = os.path.join(output_dir, "ALL_METRICS", "ALL_IMAGES")
    mask_dir = os.path.join(output_dir, "ALL_METRICS", "ALL_MASKS")
    het_dir  = os.path.join(output_dir, "ALL_METRICS", "HETEROGENEITY")
    for d in [img_dir, mask_dir, het_dir]:
        os.makedirs(d, exist_ok=True)

    clean_case_id = case_id.removesuffix('_0000') if case_id.endswith('_0000') else case_id

    # Diagnostics: the cropped CT / guide mask used as CoLIAGe input for both subregions.
    np.save(os.path.join(img_dir, f"{clean_case_id}_image.npy"), _3d_image_for_collage_yxz)
    np.save(os.path.join(mask_dir, f"{clean_case_id}_mask.npy"), _3d_mask_for_collage_yxz.astype(np.uint8))

    # =========================================================================
    # 3. CENTRAL vs. PERIPHERAL SUBREGION SPLIT (heterogeneity analysis, "13b")
    #    This IS the pipeline's output now: CoLIAGe runs ONLY on the central
    #    and peripheral subregions of the guide mask, raw (no normalization).
    #    If a subregion is empty or too small for CoLIAGe, this raises and the
    #    case is marked failed by the caller -- there is no whole-tumor
    #    fallback to save instead.
    # =========================================================================
    het = compute_central_peripheral_heterogeneity(
        ct_padded, _3d_cropped_mask_zyx.astype(bool), SVD_RADIUS, PAD_Z
    )

    label = int(is_pdac_case)
    central_vector_with_label = np.append(het['central_vector'], label).astype(np.float32)      # (27,)
    peripheral_vector_with_label = np.append(het['peripheral_vector'], label).astype(np.float32)  # (27,)
    heterogeneity_vector_with_label = np.append(het['heterogeneity_vector'], label).astype(np.float32)  # (27,)

    np.save(os.path.join(het_dir, f"{clean_case_id}_central_vector.npy"), central_vector_with_label)
    np.save(os.path.join(het_dir, f"{clean_case_id}_peripheral_vector.npy"), peripheral_vector_with_label)
    np.save(os.path.join(het_dir, f"{clean_case_id}_heterogeneity_vector.npy"), heterogeneity_vector_with_label)
    np.save(os.path.join(het_dir, f"{clean_case_id}_central_mask.npy"), het['central_mask'].astype(np.uint8))
    np.save(os.path.join(het_dir, f"{clean_case_id}_peripheral_mask.npy"), het['peripheral_mask'].astype(np.uint8))
    logger.info(
        f"Saved central/peripheral subregion radiomics for {clean_case_id} "
        f"(26 Haralick channels + PDAC label per vector, label={label})."
    )


# =========================================================================
# BATCH (ZIP) HELPERS
# =========================================================================

def find_batch_zips(data_dir):
    """
    Returns a sorted list of batch_*.zip files directly inside data_dir
    (e.g. batch_1.zip, batch_2.zip, batch_3.zip, batch_4.zip). Sorted so the
    run order is deterministic and matches the numeric suffix when present.
    """
    zips = glob.glob(os.path.join(data_dir, "batch_*.zip"))
    if not zips:
        # Fall back to ANY .zip directly in data_dir, in case the batches
        # aren't named with the batch_N.zip convention.
        zips = glob.glob(os.path.join(data_dir, "*.zip"))

    def _sort_key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        digits = ''.join(ch for ch in stem if ch.isdigit())
        return (int(digits) if digits else 0, stem)

    return sorted(zips, key=_sort_key)


def find_loose_ct_files(data_dir):
    """
    Fallback for when data_dir has NO zip files: searches data_dir
    recursively for .nii/.nii.gz files that are already sitting on disk
    (e.g. a folder of already-extracted / already-isolated cases, like a
    PDAC-only subset pulled out of the batches ahead of time). Returns the
    sorted list of CT files found, or an empty list if none exist.
    """
    ct_patterns = [
        os.path.join(data_dir, "**", "*.nii"),
        os.path.join(data_dir, "**", "*.nii.gz"),
    ]
    ct_files = []
    for pattern in ct_patterns:
        ct_files.extend(glob.glob(pattern, recursive=True))
    return sorted(ct_files)


def extract_batch_zip(zip_path, extract_root):
    """
    Extracts zip_path into extract_root/<batch_stem>/ (idempotent -- skips
    re-extraction if that folder already exists and is non-empty), then
    returns the sorted list of .nii/.nii.gz CT files found inside,
    searching recursively since files may be nested in subfolders.
    """
    batch_stem = os.path.splitext(os.path.basename(zip_path))[0]
    extract_to = os.path.join(extract_root, batch_stem)

    if os.path.isdir(extract_to) and os.listdir(extract_to):
        logger.info(f"[{batch_stem}] Already extracted at {extract_to}, skipping unzip.")
    else:
        os.makedirs(extract_to, exist_ok=True)
        logger.info(f"[{batch_stem}] Extracting {zip_path} -> {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)

    ct_patterns = [
        os.path.join(extract_to, "**", "*.nii"),
        os.path.join(extract_to, "**", "*.nii.gz"),
    ]
    ct_files = []
    for pattern in ct_patterns:
        ct_files.extend(glob.glob(pattern, recursive=True))

    logger.info(f"[{batch_stem}] Found {len(ct_files)} CT files after extraction.")
    return sorted(ct_files)


def build_label_lookup(label_dir):
    """
    Scans label_dir recursively for .nii/.nii.gz files and returns a dict
    keyed by full filename stem (same convention as the original notebook --
    no truncation here), shared across all batches.
    """
    label_patterns = [os.path.join(label_dir, "**", "*.nii"), os.path.join(label_dir, "**", "*.nii.gz")]
    label_files = []
    for pattern in label_patterns:
        label_files.extend(glob.glob(pattern, recursive=True))

    label_lookup = {}
    for p in label_files:
        key = get_full_stem(p)
        if key in label_lookup and label_lookup[key] != p:
            logger.warning(
                f"Duplicate label stem '{key}' -- '{label_lookup[key]}' is being "
                f"overwritten by '{p}'. Check your label filenames for collisions."
            )
        label_lookup[key] = p

    logger.info(f"Found {len(label_files)} label files under {label_dir} "
                f"({len(label_lookup)} unique case stems).")
    return label_lookup


def process_one_ct_file(ct_file, label_lookup, output_dir):
    """
    Resolves the label for ct_file and runs process_case(). Returns
    'success', 'skipped_no_label', or 'failed' -- never raises, so one bad
    case can't take down its batch or the overall run.
    """
    case_id, label_path = resolve_label_path(ct_file, label_lookup)

    if case_id is None:
        logger.warning(
            f"No matching label found for CT volume: {get_full_stem(ct_file)} "
            f"(tried: {get_case_id_candidates(ct_file)}). Skipping..."
        )
        return "skipped_no_label"

    if case_id != get_full_stem(ct_file):
        logger.warning(
            f"CT '{get_full_stem(ct_file)}' had no exact-stem label match; matched "
            f"via modality-suffix fallback to label case '{case_id}'. Verify this is "
            f"correct for your dataset's naming convention."
        )

    try:
        process_case(ct_file, label_path, output_dir, case_id)
        return "success"
    except Exception as e:
        logger.error(f"Failed to process case {case_id}: {str(e)}", exc_info=True)
        return "failed"


# =========================================================================
# MAIN ARGUMENT PARSING LOGIC
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process CT files for CoLIAGe central/peripheral heterogeneity analysis. "
                    "Accepts EITHER a directory of batch_*.zip files (unzipped automatically) "
                    "OR a directory that already contains loose .nii/.nii.gz files (e.g. an "
                    "already-isolated subset) -- no zip required in the latter case."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing batch_1.zip, batch_2.zip, batch_3.zip, batch_4.zip "
                              "(or any *.zip files) of CT volumes. If no zip files are found here, "
                              "this directory (searched recursively) is instead treated as a flat "
                              "folder of already-extracted .nii/.nii.gz files.")
    parser.add_argument("--label_dir", type=str, required=True,
                         help="Path to directory containing input label files, shared across all batches.")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Base output path for saving results, shared across all batches.")
    parser.add_argument("--extract_dir", type=str, default=None,
                         help="Where to unzip batches to (default: <data_dir>/_extracted).")
    args = parser.parse_args()

    extract_root = args.extract_dir or os.path.join(args.data_dir, "_extracted")
    os.makedirs(extract_root, exist_ok=True)

    batch_zips = find_batch_zips(args.data_dir)

    # (batch_stem, ct_files) pairs to process. If zips were found, each zip
    # is its own batch and gets extracted first. If NOT, fall back to
    # treating data_dir itself as one already-extracted "batch" -- this is
    # the path used for a flat folder of loose .nii.gz files (e.g. an
    # isolated PDAC-only subset), no unzip step needed.
    batches = []
    if batch_zips:
        logger.info(f"Found {len(batch_zips)} batch zip(s): {[os.path.basename(z) for z in batch_zips]}")
        for batch_zip in batch_zips:
            batch_stem = os.path.splitext(os.path.basename(batch_zip))[0]
            ct_files = extract_batch_zip(batch_zip, extract_root)
            batches.append((batch_stem, ct_files))
    else:
        logger.info(f"No zip files found in {args.data_dir} -- looking for loose .nii/.nii.gz files instead.")
        loose_files = find_loose_ct_files(args.data_dir)
        if not loose_files:
            logger.error(
                f"No batch_*.zip / *.zip files AND no .nii/.nii.gz files found under "
                f"{args.data_dir}. Nothing to process."
            )
            return
        logger.info(f"Found {len(loose_files)} loose CT file(s) directly under {args.data_dir}.")
        batches.append((os.path.basename(os.path.normpath(args.data_dir)), loose_files))

    # Label lookup is built once and shared across all batches.
    label_lookup = build_label_lookup(args.label_dir)

    grand_total = 0
    grand_success = 0
    grand_skipped = 0
    grand_failed = 0

    for batch_stem, ct_files in batches:
        logger.info("=" * 88)
        logger.info(f"BATCH START: {batch_stem}")
        logger.info("=" * 88)

        if not ct_files:
            logger.warning(f"[{batch_stem}] No .nii/.nii.gz CT files found -- skipping batch.")
            continue

        batch_success = 0
        batch_skipped = 0
        batch_failed = 0

        # Skip files that already have saved subregion output, so re-running
        # the script resumes instead of redoing finished cases.
        files_to_process = []
        for ct_file in ct_files:
            case_stem = os.path.basename(ct_file).replace("_0000.nii.gz", "").replace(".nii.gz", "")
            out_check = os.path.join(args.output_dir, 'ALL_METRICS', 'HETEROGENEITY', f"{case_stem}_heterogeneity_vector.npy")
            if not os.path.exists(out_check):
                files_to_process.append(ct_file)

        logger.info(
            f"[Filter] Of {len(ct_files)} files, {len(ct_files) - len(files_to_process)} already have "
            f"output. {len(files_to_process)} remain to be processed."
        )

        for ct_file in tqdm(files_to_process, desc=f"Processing {batch_stem}"):
            status = process_one_ct_file(ct_file, label_lookup, args.output_dir)
            if status == "success":
                batch_success += 1
            elif status == "skipped_no_label":
                batch_skipped += 1
            else:
                batch_failed += 1

        grand_total += len(ct_files)
        grand_success += batch_success
        grand_skipped += batch_skipped
        grand_failed += batch_failed

        logger.info(
            f"BATCH COMPLETE: {batch_stem} -- {batch_success}/{len(ct_files)} succeeded, "
            f"{batch_skipped} skipped (no label match), {batch_failed} failed."
        )

    logger.info("=" * 88)
    logger.info(
        f"ALL BATCHES COMPLETE: {grand_success}/{grand_total} cases succeeded across "
        f"{len(batches)} batch(es) ({grand_skipped} skipped for missing labels, "
        f"{grand_failed} failed)."
    )
    logger.info("=" * 88)


if __name__ == "__main__":
    main()
