import os
import subprocess
from glob import glob
import pydicom

TCGA_DIR = "datasets/dataset_3_TCGA-KIRC"
OUTPUT_DIR = "datasets/dataset_3_nifti"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_leaf_dicom_dirs(root_dir):
    leaf_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if any(f.endswith('.dcm') for f in files):
            leaf_dirs.append(root)
    return leaf_dirs

def convert_to_nifti():
    print("Finding TCGA-KIRC DICOM series...")
    leaf_dirs = find_leaf_dicom_dirs(TCGA_DIR)
    
    processed = 0
    for dcm_dir in leaf_dirs:
        # Find the first dicom
        dcms = glob(os.path.join(dcm_dir, "*.dcm"))
        if not dcms:
            continue
            
        try:
            ds = pydicom.dcmread(dcms[0], stop_before_pixels=True)
            patient_id = str(ds.PatientID)[:12]
            
            # Target output file: TCGA-XX-XXXX_0000.nii.gz
            # We use dcm2niix which generates: <filename>.nii.gz
            target_name = f"{patient_id}_0000"
            out_file = os.path.join(OUTPUT_DIR, f"{target_name}.nii.gz")
            
            if os.path.exists(out_file):
                # print(f"Skipping {patient_id}, already exists.")
                continue
                
            print(f"[{processed+1}] Converting TCGA patient: {patient_id}")
            
            # We want only the 3D volume, dcm2niix will auto-compress
            cmd_img = ["dcm2niix", "-z", "y", "-f", target_name, "-o", OUTPUT_DIR, dcm_dir]
            result = subprocess.run(cmd_img, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                processed += 1
            else:
                print(f"Failed {patient_id}: {result.stderr.decode()}")
                
        except Exception as e:
            print(f"Error processing dir {dcm_dir}: {e}")
            
    print(f"\\nSuccessfully formatted {processed} patients into NIfTI!")

if __name__ == "__main__":
    convert_to_nifti()
