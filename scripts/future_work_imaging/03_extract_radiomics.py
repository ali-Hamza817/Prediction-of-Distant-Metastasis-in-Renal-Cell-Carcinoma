import os
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
from glob import glob
import logging

logging.getLogger('radiomics').setLevel(logging.ERROR)

NIFTI_DIR = "datasets/dataset_3_nifti"
MASKS_DIR = "datasets/dataset_3_masks_totalseg"

def extract_features():
    print("Initializing PyRadiomics Extractor...")
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    
    results = []
    
    nifti_files = glob(os.path.join(NIFTI_DIR, "*.nii.gz"))
    print(f"Starting extraction for {len(nifti_files)} patients...")
    
    for i, img_path in enumerate(nifti_files):
        patient_id = os.path.basename(img_path).replace('_0000.nii.gz', '')
        
        # Look for the kidney mask
        tumor_mask_path = os.path.join(MASKS_DIR, f'{patient_id}_mask.nii.gz')
        
        if not os.path.exists(tumor_mask_path):
            print(f"[{i+1}/{len(nifti_files)}] Mask missing for {patient_id}, skipping.")
            continue
            
        print(f"[{i+1}/{len(nifti_files)}] Extracting {patient_id}...")
        try:
            image = sitk.ReadImage(img_path)
            mask_raw = sitk.ReadImage(tumor_mask_path)
            
            # Binarize multi-label mask to combine both kidneys into label 1
            mask = mask_raw > 0
            
            features = extractor.execute(image, mask)
            
            feat_dict = {'patient_id': patient_id}
            for key, val in features.items():
                if key.startswith('original_'):
                    feat_dict[key] = float(val)
            
            results.append(feat_dict)
            
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            
    df_res = pd.DataFrame(results)
    df_res.to_csv('datasets/dataset_3_radiomics.csv', index=False)
    print(f"Extraction complete! Saved {len(df_res)} patient features to datasets/dataset_3_radiomics.csv")

if __name__ == '__main__':
    extract_features()
