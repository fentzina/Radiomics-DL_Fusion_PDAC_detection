# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd

INPUT_DIR  = "/home/student1/ftzina_thesis/outputs/step1_res"
OUTPUT_DIR  = "/home/student1/ftzina_thesis/output_pdac"
BATCH_NAMES = ["batch_1", "batch_2", "batch_3", "batch_4"]
SECONDARY_QC_CSV = ""/home/student1/ftzina_thesis/secondary_qc_report.csv"  # Το CSV με τις 87 FAIL περιπτώσεις

X_list   = []
y_list   = []
ids_list = []

print("=" * 60)
print("Loading per-batch arrays (Way 1)...")
print("=" * 60)

for batch in BATCH_NAMES:
    x_path   = os.path.join(INPUT_DIR, f"master_X_{batch}.npy")
    y_path   = os.path.join(INPUT_DIR, f"master_y_{batch}.npy")
    ids_path = os.path.join(INPUT_DIR, f"master_ids_{batch}.npy")

    for p in [x_path, y_path, ids_path]:
        assert os.path.exists(p), f"Missing file: {p}"

    X_b   = np.load(x_path)
    y_b   = np.load(y_path)
    ids_b = np.load(ids_path, allow_pickle=True)

    X_list.append(X_b)
    y_list.append(y_b)
    ids_list.append(ids_b)
    print(f"  {batch} loaded: {X_b.shape} cases")

# ── Αρχική Ένωση Όλων των Δεδομένων στη Μνήμη ──────────────────────────────────
X_master   = np.concatenate(X_list,   axis=0)
y_master   = np.concatenate(y_list,   axis=0)
ids_master = np.concatenate(ids_list, axis=0).astype(str)

print(f"\nInitial Dataset combined in memory: {X_master.shape} cases total.")

# ── Φόρτωση των QC Φίλτρων από το CSV ─────────────────────────────────────────
if not os.path.exists(SECONDARY_QC_CSV):
    raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο {SECONDARY_QC_CSV}!")

# Διαβάζουμε το CSV χρησιμοποιώντας το ερωτηματικό ως διαχωριστικό
qc_df = pd.read_csv(SECONDARY_QC_CSV, sep=";")

# Λίστα 1: Όλα τα 87 αρχικά FAIL IDs (για το Σενάριο Α)
all_fail_ids = qc_df["case_id"].tolist()

# Λίστα 2: Μόνο τα 2 Technical Shifts (για το Σενάριο Β)
tech_shift_ids = qc_df[qc_df["secondary_verdict"] == "Technical Shift / Artifact (Exclude)"]["case_id"].tolist()

# ── ΥΛΟΠΟΙΗΣΗ ΣΕΝΑΡΙΟΥ Α (Αυστηρό Φιλτράρισμα - Αφαίρεση 87 FAIL) ───────────────
print("\n" + "-"*50)
print("GENERATING SCENARIO A (Excluding all 87 Fails)...")
indices_A = [i for i, cid in enumerate(ids_master) if cid not in all_fail_ids]

X_A = X_master[indices_A]
y_A = y_master[indices_A]
ids_A = ids_master[indices_A]

np.save(os.path.join(OUTPUT_DIR, "master_X_scenario_A.npy"), X_A.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "master_y_scenario_A.npy"), y_A.astype(np.int32))
np.save(os.path.join(OUTPUT_DIR, "master_ids_scenario_A.npy"), ids_A.astype(str))
print(f"Scenario A Saved! Total cases: {X_A.shape} (PDAC={int((y_A==1).sum())}, non-PDAC={int((y_A==0).sum())})")

# ── ΥΛΟΠΟΙΗΣΗ ΣΕΝΑΡΙΟΥ Β (Βιολογικό Φιλτράρισμα - Αφαίρεση 2 Shifts) ────────────
print("\n" + "-"*50)
print("GENERATING SCENARIO B (Keeping Exophytic, Excluding 2 Technical Shifts)...")
indices_B = [i for i, cid in enumerate(ids_master) if cid not in tech_shift_ids]

X_B = X_master[indices_B]
y_B = y_master[indices_B]
ids_B = ids_master[indices_B]

np.save(os.path.join(OUTPUT_DIR, "master_X_scenario_B.npy"), X_B.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "master_y_scenario_B.npy"), y_B.astype(np.int32))
np.save(os.path.join(OUTPUT_DIR, "master_ids_scenario_B.npy"), ids_B.astype(str))
print(f"Scenario B Saved! Total cases: {X_B.shape} (PDAC={int((y_B==1).sum())}, non-PDAC={int((y_B==0).sum())})")
print("-"*50 + "\n")

print("Both clean datasets have been successfully re-produced.")
