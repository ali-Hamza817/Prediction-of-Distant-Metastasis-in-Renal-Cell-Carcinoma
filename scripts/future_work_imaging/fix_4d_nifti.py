import os
import nibabel as nib
from glob import glob

nifti_dir = "datasets/dataset_3_nifti"
files = glob(os.path.join(nifti_dir, "*_0000.nii.gz"))

fixed_count = 0
for f in files:
    try:
        img = nib.load(f)
        if len(img.shape) > 3:
            print(f"Fixing 4D image: {f} (shape: {img.shape})")
            # Take the first 3D volume
            new_data = img.get_fdata()[..., 0]
            new_img = nib.Nifti1Image(new_data, img.affine, img.header)
            nib.save(new_img, f)
            fixed_count += 1
    except Exception as e:
        print(f"Error reading {f}: {e}")

print(f"Fixed {fixed_count} images.")
