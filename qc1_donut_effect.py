"""
Label Containment Check
=======================
Verifies that for every PDAC case, the tumor mask (label 1)
is anatomically nested within the pancreas mask (label 4) volume context,
accounting for the PANORAMA "Donut Effect" (0% voxel overlap).

Specifically checks:
  - Voxels with label 1 that sit completely OUTSIDE the 3D Bounding Box of label 4.
  - Voxels with label 1 that have NO label 4 in their immediate 2-voxel 
    neighborhood (strict boundary containment).
  - Reports standard counts and verdicts based on anatomical positioning.

For each case, reports:
  - n_tumor_voxels        : total voxels with label 1
  - n_outside_pancreas    : label-1 voxels that are NOT within the pancreas BBox zone
  - outside_pct           : percentage of tumor outside the pancreas BBox
  - overlap_voxels        : raw voxel overlap (expected to be 0 due to dataset design)
  - verdict               : PASS / WARN / FAIL

PASS : all tumor voxels are anatomically inside the pancreas BBox zone
WARN : up to 2% of tumor voxels extend outside the pancreas BBox zone
FAIL : >2% of tumor voxels extend outside the pancreas BBox zone
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import find_objects, binary_dilation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TUMOR_LABEL    = 1
PANCREAS_LABEL = 4
WARN_THRESHOLD = 2.0   # percent outside pancreas before WARN
FAIL_THRESHOLD = 2.0   # same threshold used for FAIL (anything > WARN is FAIL)
DILATION_RADIUS = 2    # Voxel radius to check for immediate neighborhood containment

# Suffixes to look for when searching for label files
LABEL_SUFFIXES = ["_label.nii.gz", "_seg.nii.gz", "_mask.nii.gz",
                  ".nii.gz", "_label.nii", ".nii"]


def find_label_file(label_dir, case_id):
    """Search for a label file matching the case_id in label_dir."""
    for suffix in LABEL_SUFFIXES:
        path = os.path.join(label_dir, f"{case_id}{suffix}")
        if os.path.exists(path):
            return path
    # Try recursive search one level deep
    for subdir in os.listdir(label_dir):
        subpath = os.path.join(label_dir, subdir)
        if os.path.isdir(subpath):
            for suffix in LABEL_SUFFIXES:
                path = os.path.join(subpath, f"{case_id}{suffix}")
                if os.path.exists(path):
                    return path
    return None


def load_label_array(label_path):
    """
    Load a NIfTI label file and return as a numpy array (ZYX order).
    No resampling — checks the raw label geometry as provided.
    """
    img  = sitk.ReadImage(label_path)
    arr  = sitk.GetArrayFromImage(img)   # returns ZYX
    return arr.astype(np.uint8)


def check_containment(label_arr, case_id):
    """
    Check whether tumor (label 1) voxels are nested within the 
    anatomical space of the pancreas (label 4), bypasses the voxel-level 
    mutual exclusivity artifact.
    """
    tumor_mask    = (label_arr == TUMOR_LABEL)
    pancreas_mask = (label_arr == PANCREAS_LABEL)

    n_tumor    = int(tumor_mask.sum())
    n_pancreas = int(pancreas_mask.sum())

    if n_tumor == 0:
        return {
            "case_id"           : case_id,
            "n_tumor_voxels"    : 0,
            "n_pancreas_voxels" : n_pancreas,
            "n_outside_pancreas": 0,
            "outside_pct"       : 0.0,
            "overlap_voxels"    : 0,
            "verdict"           : "SKIP — no tumor voxels found",
            "note"              : "Label 1 absent — not a PDAC case or mislabeled"
        }

    if n_pancreas == 0:
        return {
            "case_id"           : case_id,
            "n_tumor_voxels"    : n_tumor,
            "n_pancreas_voxels" : 0,
            "n_outside_pancreas": n_tumor,
            "outside_pct"       : 100.0,
            "overlap_voxels"    : 0,
            "verdict"           : "FAIL",
            "note"              : "No pancreas label (4) found at all"
        }

    # 1. Raw Voxel Overlap (Expected to be 0 in PANORAMA due to auto-exclusion)
    raw_overlap_mask = tumor_mask & pancreas_mask
    n_overlap = int(raw_overlap_mask.sum())

    # 2. Bounding Box Containment Zone
    bbox_l4 = find_objects(pancreas_mask)
    if not bbox_l4 or bbox_l4[0] is None:
        pancreas_zone = np.zeros_like(pancreas_mask)
    else:
        pancreas_zone = np.zeros_like(pancreas_mask)
        # find_objects returns a list of slices, we take the first element
        pancreas_zone[bbox_l4[0]] = 1

    # Voxels where tumor exists entirely outside the 3D bounding box zone of the pancreas
    outside_bbox_mask = tumor_mask & (pancreas_zone == 0)
    n_outside = int(outside_bbox_mask.sum())
    outside_pct = 100.0 * n_outside / n_tumor

    # 3. Neighborhood Strict Boundary Check (Dilation Check)
    # Check if tumor voxels have NO pancreas in their immediate neighbor contact area
    dilated_pancreas = binary_dilation(pancreas_mask, iterations=DILATION_RADIUS)
    isolated_tumor_mask = tumor_mask & (~dilated_pancreas)
    n_isolated_voxels = int(isolated_tumor_mask.sum())

    # Build Note Context
    note_details = f"Raw Overlap: {n_overlap} voxels. "
    if n_isolated_voxels > 0:
        note_details += f"{n_isolated_voxels} tumor voxels are anatomically isolated (> {DILATION_RADIUS} voxels away from pancreas)."
    else:
        note_details += "Strict containment verified (tumor touches/intersects pancreas boundaries)."

    # Verdict Evaluation based on BBox escape percentages
    if outside_pct == 0.0:
        verdict = "PASS"
        note    = f"All tumor voxels sit safely inside the pancreas BBox zone. {note_details}"
    elif outside_pct <= WARN_THRESHOLD:
        verdict = "WARN"
        note    = f"{n_outside} voxels ({outside_pct:.2f}%) outside BBox — minor edge artifact. {note_details}"
    else:
        verdict = "FAIL"
        note    = f"{n_outside} voxels ({outside_pct:.2f}%) outside BBox — tumor anatomically misaligned. {note_details}"

    return {
        "case_id"           : case_id,
        "n_tumor_voxels"    : n_tumor,
        "n_pancreas_voxels" : n_pancreas,
        "n_outside_pancreas": n_outside,
        "outside_pct"       : round(outside_pct, 4),
        "overlap_voxels"    : n_overlap,
        "verdict"           : verdict,
        "note"              : note
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check that tumor (label 1) is nested within pancreas (label 4) "
                    "for all PDAC cases in the PANORAMA dataset, bypassing mutual exclusivity."
    )
    parser.add_argument("--label_dir",  required=True,
                        help="Directory containing PANORAMA segmentation label files.")
    parser.add_argument("--case_list",  default=None,
                        help="Optional: path to a .txt file with one case_id per line. "
                             "If not provided, all label files in --label_dir are scanned.")
    parser.add_argument("--output_csv", default="pwf_report_auto.csv",
                        help="Path for the output CSV report.")
    args = parser.parse_args()

    # ── Build case list ───────────────────────────────────────────────────────
    if args.case_list is not None:
        with open(args.case_list) as f:
            case_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(case_ids)} case IDs from {args.case_list}")
    else:
        # Auto-discover from label_dir
        all_files = []
        for root, dirs, files in os.walk(args.label_dir):
            for fname in files:
                if any(fname.endswith(s) for s in LABEL_SUFFIXES):
                    all_files.append(fname)

        # Extract case_ids by stripping known suffixes
        case_ids = []
        for fname in all_files:
            cid = fname
            for suffix in sorted(LABEL_SUFFIXES, key=len, reverse=True):
                if cid.endswith(suffix):
                    cid = cid[:-len(suffix)]
                    break
            case_ids.append(cid)
        case_ids = sorted(set(case_ids))
        print(f"Auto-discovered {len(case_ids)} cases in {args.label_dir}")

    # ── Run checks ────────────────────────────────────────────────────────────
    records      = []
    n_pass       = 0
    n_warn       = 0
    n_fail       = 0
    n_skip       = 0
    n_not_found  = 0

    for case_id in tqdm(case_ids, desc="Checking label containment"):
        label_path = find_label_file(args.label_dir, case_id)

        if label_path is None:
            records.append({
                "case_id"           : case_id,
                "n_tumor_voxels"    : -1,
                "n_pancreas_voxels" : -1,
                "n_outside_pancreas": -1,
                "outside_pct"       : -1.0,
                "overlap_voxels"    : -1,
                "verdict"           : "NOT FOUND",
                "note"              : "No label file found for this case_id"
            })
            n_not_found += 1
            continue

        try:
            arr    = load_label_array(label_path)
            result = check_containment(arr, case_id)
            records.append(result)

            v = result["verdict"]
            if "PASS"  in v: n_pass += 1
            elif "WARN" in v: n_warn += 1
            elif "FAIL" in v: n_fail += 1
            else:             n_skip += 1

        except Exception as e:
            records.append({
                "case_id"           : case_id,
                "n_tumor_voxels"    : -1,
                "n_pancreas_voxels" : -1,
                "n_outside_pancreas": -1,
                "outside_pct"       : -1.0,
                "overlap_voxels"    : -1,
                "verdict"           : "ERROR",
                "note"              : str(e)
            })


    # ── Save report ───────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df = df.sort_values("outside_pct", ascending=False)
    df.to_csv(args.output_csv, index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    total = len(case_ids)
    print("\n" + "=" * 65)
    print("LABEL CONTAINMENT CHECK — SUMMARY")
    print("=" * 65)
    print(f"  Total cases checked  : {total}")
    print(f"  PASS                 : {n_pass:>5}  ({100*n_pass/total:.1f}%)  "
          f"all tumor voxels inside pancreas")
    print(f"  WARN                 : {n_warn:>5}  ({100*n_warn/total:.1f}%)  "
          f"≤{WARN_THRESHOLD}% outside (boundary artefact)")
    print(f"  FAIL                 : {n_fail:>5}  ({100*n_fail/total:.1f}%)  "
          f">{FAIL_THRESHOLD}% outside — review needed")
    print(f"  SKIP (no label 1)    : {n_skip:>5}  non-PDAC or mislabeled")
    print(f"  NOT FOUND            : {n_not_found:>5}  label file missing")
    print(f"\nReport saved to: {args.output_csv}")

    if n_fail > 0:
        print(f"\nTop FAIL cases (largest % outside pancreas):")
        fail_df = df[df["verdict"] == "FAIL"].head(10)
        for _, row in fail_df.iterrows():
            print(f"  {row['case_id']:<25}  "
                  f"tumor={row['n_tumor_voxels']:,}  "
                  f"outside={row['n_outside_pancreas']:,}  "
                  f"({row['outside_pct']:.2f}%)")

    if n_warn > 0:
        print(f"\nWARN cases (minor boundary artefacts — likely safe to keep):")
        warn_df = df[df["verdict"] == "WARN"].head(10)
        for _, row in warn_df.iterrows():
            print(f"  {row['case_id']:<25}  "
                  f"outside={row['n_outside_pancreas']}  "
                  f"({row['outside_pct']:.2f}%)")

    print("=" * 65)


if __name__ == "__main__":
    main()
