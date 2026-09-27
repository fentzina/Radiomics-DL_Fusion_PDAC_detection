"""
AGGREGATION STEP B
Stacks per-case central/peripheral/heterogeneity vectors into cohort-level matrices for the heterogeneity analysis.
Run once per internal_fraction AFTER the pipeline completes.
"""

import os
import numpy as np

# CONFIG — change FRACTION and HETERO_DIR for each run
FRACTION   = 0.6    # change to 0.4 or 0.6 for the other runs

HETERO_DIR = f"/home/student1/ftzina_thesis/outputs/subregions_tumor_only/ALL_METRICS_0.6/HETEROGENEITY"
OUTPUT_DIR = f"/home/student1/ftzina_thesis/outputs/subregions_tumor_only/fraction_0.6_aggregated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Discover all central vector files — these are the anchor
# ─────────────────────────────────────────────────────────────────────────────
central_files = sorted([
    f for f in os.listdir(HETERO_DIR)
    if f.endswith("_central_vector.npy")
])

print(f"Found {len(central_files)} central vector files for fraction={FRACTION}")

records_central    = []
records_peripheral = []
records_hetero     = []
case_ids           = []
skipped            = []

for fname in central_files:
    case_id = fname.replace("_central_vector.npy", "")

    central_path    = os.path.join(HETERO_DIR, f"{case_id}_central_vector.npy")
    peripheral_path = os.path.join(HETERO_DIR, f"{case_id}_peripheral_vector.npy")
    hetero_path     = os.path.join(HETERO_DIR, f"{case_id}_heterogeneity_vector.npy")

    # All three must exist
    if not all(os.path.exists(p) for p in [central_path, peripheral_path, hetero_path]):
        print(f"  SKIP {case_id} — missing one or more vector files")
        skipped.append(case_id)
        continue

    # Load — shape is (27,): 26 Haralick means + label in last position
    c_vec  = np.load(central_path).astype(np.float32)
    p_vec  = np.load(peripheral_path).astype(np.float32)
    h_vec  = np.load(hetero_path).astype(np.float32)

    # Sanity check
    if c_vec.shape[0] != 27 or p_vec.shape[0] != 27 or h_vec.shape[0] != 27:
        print(f"  SKIP {case_id} — unexpected vector shape "
              f"c={c_vec.shape} p={p_vec.shape} h={h_vec.shape}")
        skipped.append(case_id)
        continue

    # Verify it is a PDAC case (label should be 1 in last position)
    label = int(round(c_vec[-1]))
    if label != 1:
        print(f"  SKIP {case_id} — label={label} (not PDAC)")
        skipped.append(case_id)
        continue

    # Keep only the 26 Haralick channels (drop the label column)
    records_central.append(c_vec[:26])
    records_peripheral.append(p_vec[:26])
    records_hetero.append(h_vec[:26])
    case_ids.append(case_id)

print(f"\nSuccessfully loaded : {len(records_central)} cases")
print(f"Skipped             : {len(skipped)} cases")

if len(records_central) == 0:
    raise RuntimeError("No cases loaded — check HETERO_DIR path.")

# ─────────────────────────────────────────────────────────────────────────────
# Stack into cohort-level matrices
# ─────────────────────────────────────────────────────────────────────────────
X_central    = np.stack(records_central,    axis=0)  # (N, 26)
X_peripheral = np.stack(records_peripheral, axis=0)  # (N, 26)
X_hetero     = np.stack(records_hetero,     axis=0)  # (N, 26)
ids_array    = np.array(case_ids, dtype=str)

print(f"\nMatrix shapes:")
print(f"  X_central    : {X_central.shape}")
print(f"  X_peripheral : {X_peripheral.shape}")
print(f"  X_hetero     : {X_hetero.shape}")

# Sanity checks
assert not np.isnan(X_central).any(),    "NaNs in central matrix"
assert not np.isnan(X_peripheral).any(), "NaNs in peripheral matrix"
assert X_central.shape == X_peripheral.shape == X_hetero.shape

# Save
np.save(os.path.join(OUTPUT_DIR, "X_central.npy"),    X_central)
np.save(os.path.join(OUTPUT_DIR, "X_peripheral.npy"), X_peripheral)
np.save(os.path.join(OUTPUT_DIR, "X_hetero.npy"),     X_hetero)
np.save(os.path.join(OUTPUT_DIR, "case_ids.npy"),     ids_array)

print(f"\nSaved to: {OUTPUT_DIR}")
print(f"  X_central.npy    → {X_central.shape}")
print(f"  X_peripheral.npy → {X_peripheral.shape}")
print(f"  X_hetero.npy     → {X_hetero.shape}")
print(f"  case_ids.npy     → {ids_array.shape}")
print(f"\nAggregation complete for fraction={FRACTION}.")
print(f"Run heterogeneity_analysis.py next, pointing INPUT_DIR to:")
print(f"  {OUTPUT_DIR}")
