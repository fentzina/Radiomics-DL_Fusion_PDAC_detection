#!/usr/bin/env python3
import os
import shutil

# =========================================================================
# CONFIG
# =========================================================================
VALID_IDS_FILE = "/home/student1/ftzina_thesis/output_pdac/filtered_pdac_cases_ids.txt"
SOURCE_DIR     = "/home/student1/ftzina_thesis/outputs/subregions_tumor_only/ALL_METRICS_0.5/HETEROGENEITY_0.5"
TARGET_DIR     = "/home/student1/ftzina_thesis/output_pdac/intra_tumor_589_PDACs_0.5"

def main():
    print(f"[*] Reading valid IDs from: {VALID_IDS_FILE}")
    if not os.path.exists(VALID_IDS_FILE):
        print(f"[!] Error: {VALID_IDS_FILE} not found.")
        return

    # Διάβασμα των 589 καθαρών IDs
    with open(VALID_IDS_FILE, "r") as f:
        valid_ids = set(line.strip() for line in f if line.strip())

    print(f"[*] Loaded {len(valid_ids)} target case IDs from txt file.")

    # Δημιουργία του νέου φακέλου αν δεν υπάρχει
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"[*] Target directory set to: {TARGET_DIR}")

    # Τα 5 είδη αρχείων που ψάχνουμε για κάθε case
    suffixes = [
        "_central_mask.npy",
        "_peripheral_mask.npy",
        "_central_vector.npy",
        "_peripheral_vector.npy",
        "_heterogeneity_vector.npy"
    ]

    copied_count = 0
    missing_cases = []

    print("\n[*] Starting isolation process...")
    
    # Έλεγχος για κάθε ένα από τα 589 IDs
    for case_id in sorted(valid_ids):
        # Δοκιμάζουμε τόσο το σκέτο ID όσο και το ID με το επίθεμα _00001
        found_case = False
        
        # Ελέγχουμε ποια μορφή αρχείου υπάρχει στον δίσκο
        # (π.χ. 102216_central_mask.npy ή 102216_00001_central_mask.npy)
        possible_prefixes = [case_id, f"{case_id}_00001"]
        
        chosen_prefix = None
        for prefix in possible_prefixes:
            test_file = os.path.join(SOURCE_DIR, f"{prefix}{suffixes[0]}")
            if os.path.exists(test_file):
                chosen_prefix = prefix
                break
        
        if chosen_prefix:
            # Αν βρέθηκε η σωστή μορφή, αντιγράφουμε και τα 5 αρχεία
            all_5_exist = True
            for suff in suffixes:
                src_file = os.path.join(SOURCE_DIR, f"{chosen_prefix}{suff}")
                if not os.path.exists(src_file):
                    all_5_exist = False
                    print(f"[!] Missing sub-file: {src_file}")
                    continue
                
                # Αντιγραφή στον νέο φάκελο
                dst_file = os.path.join(TARGET_DIR, f"{chosen_prefix}{suff}")
                shutil.copy2(src_file, dst_file)
                copied_count += 1
            
            if all_5_exist:
                found_case = True
        
        if not found_case:
            missing_cases.append(case_id)

    # =========================================================================
    # ΣΤΑΤΙΣΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ
    # =========================================================================
    print("\n" + "="*60)
    print("PROCESS COMPLETED")
    print("="*60)
    print(f"Expected cases   : {len(valid_ids)}")
    print(f"Expected files   : {len(valid_ids) * 5}")
    print(f"Successfully moved files: {copied_count} (out of {len(valid_ids)*5})")
    print(f"Successfully isolated: {copied_count // 5} / {len(valid_ids)} cases")
    
    if missing_cases:
        print(f"\n[!] Warning: {len(missing_cases)} cases were NOT found in the source directory:")
        for mc in missing_cases:
            print(f"  - {mc}")
    else:
        print("\n[+] Success! All 2,945 files were perfectly isolated with no missing cases.")
    print("="*60)

if __name__ == "__main__":
    main()
