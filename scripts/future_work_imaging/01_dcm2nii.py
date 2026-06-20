import os
import subprocess
from glob import glob

DICOM_DIR = "datasets/dataset_3_TCGA-KIRC"
OUTPUT_DIR = "datasets/dataset_3_nifti"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_leaf_dicom_dirs(root_dir):
    leaf_dirs = []
    for root, dirs, files in os.walk(root_dir):
        # If there are DICOM files in this directory
        if any(f.endswith('.dcm') for f in files):
            leaf_dirs.append(root)
    return leaf_dirs

if __name__ == "__main__":
    print(f"Scanning {DICOM_DIR} for DICOM series...")
    leaf_dirs = find_leaf_dicom_dirs(DICOM_DIR)
    print(f"Found {len(leaf_dirs)} DICOM series.")

    processed = 0
    for dcm_dir in leaf_dirs:
        # Extract patient ID (e.g. TCGA-BP-4162) from path
        parts = dcm_dir.split(os.sep)
        patient_id = None
        for p in parts:
            if p.startswith('TCGA-'):
                patient_id = p
                break
        
        if not patient_id:
            continue
            
        # We only want the first 12 chars: TCGA-XX-XXXX
        patient_id = patient_id[:12]
        
        # dcm2niix output format: %i (ID)
        cmd = [
            "dcm2niix",
            "-z", "y", # Compress to .nii.gz
            "-f", patient_id,
            "-o", OUTPUT_DIR,
            dcm_dir
        ]
        
        print(f"Converting {patient_id}...")
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processed += 1
        except Exception as e:
            print(f"Failed to convert {patient_id}: {e}")
            
    print(f"Successfully converted {processed} DICOM series to NIfTI.")
