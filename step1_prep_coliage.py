#!/usr/bin/env python3
"""
Automated 3D CT Preprocessing & CoLIAGe Feature Extraction Pipeline -- batch mode.

This is fml_fixed_NEW.py with the exact same per-case processing logic
(helper functions, Collage class, process_case -- byte-for-byte unchanged),
but with a batch driver on top that automatically works through several
zipped batches of CT scans instead of a single --ct_dir folder.

Usage:
    python batch_coliage_pipeline_FIXED.py \
        --data_dir ./data \
        --label_dir ./panorama_cases \
        --output_dir ./outputs

Expects --data_dir to contain one or more batch_*.zip files (e.g. batch_1.zip,
batch_2.zip, batch_3.zip, batch_4.zip), each containing .nii/.nii.gz CT volumes
(possibly nested in subfolders inside the zip). Every batch is unzipped,
scanned for CT volumes, and processed against the single shared --label_dir,
with output written into a common --output_dir. One CT file failing to
process does not stop the rest of the batch (or the other batches).
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


def normalize_volume_channelwise(tensor_zyx_c, mask_3d):
    normalized = np.zeros_like(tensor_zyx_c, dtype=np.float32)
    for c in range(tensor_zyx_c.shape[-1]):
        ch = tensor_zyx_c[:, :, :, c]
        valid = mask_3d & np.isfinite(ch)
        tissue_vals = ch[valid]
        if len(tissue_vals) == 0 or tissue_vals.max() == tissue_vals.min():
            continue
        v_min = tissue_vals.min()
        v_max = tissue_vals.max()
        normalized[:, :, :, c] = np.where(
            valid,
            (ch - v_min) / (v_max - v_min + 1e-8),
            0.0
        )
    return normalized


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
    #tumor_3d_mask = (resampled_labels_numpy == TUMOR_LABEL) #combined_mask_labels_1_4_3d = tumor_3d_mask | pancreas_3d_mask
    pancreas_3d_mask = (resampled_labels_numpy == PANCREAS_LABEL)
    
    # Define Crop Coordinates
    guide_mask = pancreas_3d_mask.astype(bool) # guide_mask = combined_mask_labels_1_4_3d.astype(bool) if is_pdac_case else pancreas_3d_mask.astype(bool)
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
    # 2. PAD THE 128^3 CROP, THEN RUN COLIAGE ON THE PADDED CROP
    #    Array axis order here is (Y, X, Z). The SVD dominant-orientation window
    #    is 2D over (Y, X) with diameter 2*SVD_RADIUS+1, so BOTH Y and X need a
    #    full SVD_RADIUS margin to avoid boundary clamping. Z only needs a
    #    1-slice margin (PAD_Z) for the finite-difference gradient at the
    #    volume edge. (The original notebook/earlier fml.py padded axis0 by
    #    PAD_Z=1 and axis2 by SVD_RADIUS -- i.e. gave Z more margin than it
    #    needs and starved Y of the margin it actually needs. That is harmless
    #    when the guide mask's Y-extent happens to sit comfortably inside the
    #    128 crop, but crashes with a broadcast shape error whenever the mask
    #    gets close to the crop's Y boundary, which is exactly what happened
    #    on case 100029_00001_0000.)
    # =========================================================================
    logger.info("Executing 3D CoLIAGe on the padded 128^3 crop (matches notebook order)...")

    SVD_RADIUS = 3
    PAD_Z = 1  # gradient edge margin, applies to the Z axis only

    ct_padded = np.pad(
        _3d_image_for_collage_yxz,
        ((SVD_RADIUS, SVD_RADIUS), (SVD_RADIUS, SVD_RADIUS), (PAD_Z, PAD_Z)),
        mode='reflect',
    )
    mask_padded = np.pad(
        _3d_mask_for_collage_yxz.astype(np.uint8),
        ((SVD_RADIUS, SVD_RADIUS), (SVD_RADIUS, SVD_RADIUS), (PAD_Z, PAD_Z)),
        mode='constant', constant_values=0,
    )

    collage_3d_instance = Collage(
        ct_padded,
        mask_padded,
        svd_radius=SVD_RADIUS,
        num_unique_angles=32,
    )
    collage_3d_instance.execute()
    logger.info(f"Raw padded CoLIAGe output shape: {collage_3d_instance.collage_output.shape}")

    # Trim the padding back out -> original (Y, X, Z) = 128^3 spatial size
    raw = collage_3d_instance.collage_output
    raw_trimmed = raw[SVD_RADIUS:-SVD_RADIUS, SVD_RADIUS:-SVD_RADIUS, PAD_Z:-PAD_Z, :, :]


    # Transpose back from YXZ to ZYX: (Y, X, Z, 13, 2) -> (Z, Y, X, 13, 2)
    _3d_haralick_volume_zyx = np.transpose(raw_trimmed, (2, 0, 1, 3, 4))

    # Flatten 13 features x 2 angles -> 26 channels: (128, 128, 128, 26)
    Z, Y, X = _3d_haralick_volume_zyx.shape[:3]
    _3d_haralick_26ch = _3d_haralick_volume_zyx.reshape(Z, Y, X, -1)

    if _3d_haralick_volume_zyx.shape != (Z, Y, X, 13, 2):
        raise ValueError(f"Unexpected Haralick shape: {_3d_haralick_volume_zyx.shape}")
    if _3d_haralick_26ch.shape != (Z, Y, X, 26):
        raise ValueError(f"Unexpected 26ch shape: {_3d_haralick_26ch.shape}")
    if np.all(np.isnan(_3d_haralick_volume_zyx)):
        raise ValueError("Haralick output is entirely NaN -- mask was empty or CoLIAGe failed")

    # =========================================================================
    # 3. CONCATENATE AND SAVE (raw, unnormalized)
    # =========================================================================
    # Build 27-channel raw tensor: 1 CT channel + 26 Haralick channels
    ct_channel = _3d_cropped_ct_zyx[:, :, :, np.newaxis]
    tensor_raw = np.concatenate([ct_channel, _3d_haralick_26ch], axis=-1)

    # Append the case label as channel 28 (no normalization -- raw CT + raw radiomics)
    label_channel = np.full(tensor_raw.shape[:-1] + (1,), int(is_pdac_case), dtype=np.float32)
    tensor_with_label = np.concatenate([tensor_raw, label_channel], axis=-1)

    # Create subdirectories and write out matrix files
    npy_dir  = os.path.join(output_dir, "ALL_TENSORS", "npy_files")
    img_dir  = os.path.join(output_dir, "ALL_METRICS", "ALL_IMAGES")
    mask_dir = os.path.join(output_dir, "ALL_METRICS", "ALL_MASKS")
    for d in [npy_dir, img_dir, mask_dir]:
        os.makedirs(d, exist_ok=True)

    clean_case_id = case_id.removesuffix('_0000') if case_id.endswith('_0000') else case_id     #clean_case_id = case_id.replace('_0000', '')
    tensor_out_path = os.path.join(npy_dir, f"{clean_case_id}_tensor_128_27ch.npy")
    np.save(tensor_out_path, tensor_with_label.astype(np.float32))
    logger.info(f"Successfully saved raw (unnormalized) 28-channel tensor -> {tensor_out_path}")

    # Save additional diagnostics matching original outputs
    # (_3d_image_for_collage_yxz / _3d_mask_for_collage_yxz already computed above, step 1)
    np.save(os.path.join(img_dir, f"{clean_case_id}_image.npy"), _3d_image_for_collage_yxz)
    np.save(os.path.join(mask_dir, f"{clean_case_id}_mask.npy"), _3d_mask_for_collage_yxz.astype(np.uint8))


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
        description="Automatically unzip and process several batches of CT files for CoLIAGe feature generation."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing batch_1.zip, batch_2.zip, batch_3.zip, batch_4.zip "
                              "(or any *.zip files) of CT volumes.")
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
    if not batch_zips:
        logger.error(f"No batch_*.zip (or *.zip) files found in {args.data_dir}. Nothing to process.")
        return
    logger.info(f"Found {len(batch_zips)} batch zip(s): {[os.path.basename(z) for z in batch_zips]}")

    # Label lookup is built once and shared across all batches.
    label_lookup = build_label_lookup(args.label_dir)

    grand_total = 0
    grand_success = 0
    grand_skipped = 0
    grand_failed = 0

    for batch_zip in batch_zips:
        batch_stem = os.path.splitext(os.path.basename(batch_zip))[0]
        logger.info("=" * 88)
        logger.info(f"BATCH START: {batch_stem} ({batch_zip})")
        logger.info("=" * 88)

        ct_files = extract_batch_zip(batch_zip, extract_root)
        if not ct_files:
            logger.warning(f"[{batch_stem}] No .nii/.nii.gz CT files found after extraction -- skipping batch.")
            continue

        batch_success = 0
        batch_skipped = 0
        batch_failed = 0

        #############################################################
        #for ct_file in tqdm(ct_files, desc=f"Processing {batch_stem}"): #for ct_file in ct_files:
        #    status = process_one_ct_file(ct_file, label_lookup, args.output_dir)
        #    if status == "success":
        #        batch_success += 1
        #    elif status == "skipped_no_label":
        #        batch_skipped += 1
        #    else:
        #        batch_failed += 1
        
        # 1. Κρατάμε ΜΟΝΟ τα αρχεία που ΔΕΝ έχουν ήδη παραχθεί
        files_to_process = []
        for ct_file in ct_files:
            # Παίρνουμε το καθαρό case_stem (το λυμένο από χθες)
            case_stem = os.path.basename(ct_file).replace("_0000.nii.gz", "").replace(".nii.gz", "")
            out_check = os.path.join(args.output_dir, 'ALL_TENSORS', 'npy_files', f"{case_stem}_tensor_128_27ch.npy")
            
            if not os.path.exists(out_check):
                files_to_process.append(ct_file)

        logger.info(f"[Filter] Από τα {len(ct_files)} αρχεία, τα {len(ct_files) - len(files_to_process)} υπάρχουν ήδη. Ξεκινάνε τα {len(files_to_process)} που απομένουν.")

        # 2. Το loop τρέχει ΜΟΝΟ για όσα λείπουν — Η μπάρα δείχνει την ΠΡΑΓΜΑΤΙΚΗ πρόοδο
        for ct_file in tqdm(files_to_process, desc=f"Processing {batch_stem}"):
            status = process_one_ct_file(ct_file, label_lookup, args.output_dir)
            if status == "success":
                batch_success += 1
            elif status == "skipped_no_label":
                batch_skipped += 1
            else:
                batch_failed += 1
##################################################################################

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
        f"{len(batch_zips)} batch(es) ({grand_skipped} skipped for missing labels, "
        f"{grand_failed} failed)."
    )
    logger.info("=" * 88)


if __name__ == "__main__":
    main()
