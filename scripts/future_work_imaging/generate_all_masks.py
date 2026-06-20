import os
import subprocess
from glob import glob

NIFTI_DIR = "datasets/dataset_3_nifti"
MASKS_DIR = "datasets/dataset_3_masks_totalseg"

os.makedirs(MASKS_DIR, exist_ok=True)

def generate_masks():
    print("Generating Whole-Kidney Masks using TotalSegmentator...")
    nifti_files = sorted(glob(os.path.join(NIFTI_DIR, "*_0000.nii.gz")))
    
    processed = 0
    for img_path in nifti_files:
        filename = os.path.basename(img_path)
        patient_id = filename.split("_0000.nii.gz")[0]
        
        mask_path = os.path.join(MASKS_DIR, f"{patient_id}_mask.nii.gz")
        
        if os.path.exists(mask_path):
            continue
            
        print(f"[{processed+1}] Segmenting: {patient_id}")
        
        # We run the fast model to save time (3mm resolution)
        # We extract only the kidneys as the ROI subset to save memory and time
        cmd = [
            "TotalSegmentator",
            "-i", img_path,
            "-o", mask_path,
            "-ml", # Save multi-label (one file)
            "-rs", "kidney_left", "kidney_right",
            "--fast"
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode == 0:
            processed += 1
            print(f"  -> Success: {mask_path}")
        else:
            err_msg = result.stderr.decode() if result.stderr else "Unknown error"
            print(f"  -> Failed: {err_msg}")

    print(f"\\nSuccessfully generated masks for {processed} patients!")

if __name__ == "__main__":
    generate_masks()
