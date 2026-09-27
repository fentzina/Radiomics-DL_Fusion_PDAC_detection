import numpy as np
import os
import glob

het_dir = "/home/student1/ftzina_thesis/outputs/heterogen/ALL_METRICS/HETEROGENEITY"

case_ids         = []
central_vectors  = []
peripheral_vectors = []
heterogeneity_vectors = []

# Find all cases by looking for heterogeneity_vector files
het_files = sorted(glob.glob(os.path.join(het_dir, "*_heterogeneity_vector.npy")))

for hf in het_files:
    case_id = os.path.basename(hf).replace("_heterogeneity_vector.npy", "")
    c_path  = os.path.join(het_dir, f"{case_id}_central_vector.npy")
    p_path  = os.path.join(het_dir, f"{case_id}_peripheral_vector.npy")

    if not os.path.exists(c_path) or not os.path.exists(p_path):
        print(f"WARNING: missing central or peripheral vector for {case_id}, skipping.")
        continue

    case_ids.append(case_id)
    central_vectors.append(np.load(c_path))
    peripheral_vectors.append(np.load(p_path))
    heterogeneity_vectors.append(np.load(hf))

central_vectors       = np.array(central_vectors)        # (371, 26)
peripheral_vectors    = np.array(peripheral_vectors)     # (371, 26)
heterogeneity_vectors = np.array(heterogeneity_vectors)  # (371, 26)
case_ids              = np.array(case_ids)

print(f"Loaded {len(case_ids)} cases")
print(f"Central vectors shape    : {central_vectors.shape}")
print(f"Peripheral vectors shape : {peripheral_vectors.shape}")
print(f"Heterogeneity vectors shape: {heterogeneity_vectors.shape}")

# Save aggregated arrays for the analysis script
out_dir = "/home/student1/ftzina_thesis/outputs/heterogen/aggregated"
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "central_vectors.npy"),       central_vectors.astype(np.float32))
np.save(os.path.join(out_dir, "peripheral_vectors.npy"),    peripheral_vectors.astype(np.float32))
np.save(os.path.join(out_dir, "heterogeneity_vectors.npy"), heterogeneity_vectors.astype(np.float32))
np.save(os.path.join(out_dir, "case_ids.npy"),              case_ids.astype(str))
print(f"Saved aggregated arrays to {out_dir}")
