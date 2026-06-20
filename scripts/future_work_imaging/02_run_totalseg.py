import os
import subprocess
from glob import glob

NIFTI_DIR = "datasets/dataset_3_nifti"
MASKS_DIR = "datasets/dataset_3_masks"

os.makedirs(MASKS_DIR, exist_ok=True)

if __name__ == "__main__":
    nifti_files = glob(os.path.join(NIFTI_DIR, "*.nii.gz"))
    print(f"Found {len(nifti_files)} NIfTI files to segment.")
    
    for i, file_path in enumerate(nifti_files):
        # Patient ID is the filename
        patient_id = os.path.basename(file_path).replace('.nii.gz', '')
        output_patient_dir = os.path.join(MASKS_DIR, patient_id)
        
        if os.path.exists(output_patient_dir):
            print(f"[{i+1}/{len(nifti_files)}] Skipping {patient_id}, already segmented.")
            continue
            
        print(f"[{i+1}/{len(nifti_files)}] Segmenting {patient_id}...")
        
        # Run TotalSegmentator with kidney_tumor task
        # It will output kidney.nii.gz and kidney_tumor.nii.gz in the output folder
        cmd = [
            "TotalSegmentator",
            "-i", file_path,
            "-o", output_patient_dir,
            "--task", "kidney_tumor"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"TotalSegmentator failed on {patient_id}: {e}")
            
    print("Auto-segmentation job completed!")
