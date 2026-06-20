#!/bin/bash
set -e

# Setup Environment
echo "=== STEP 1: Setting up Python 3.9 Environment ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n seg_env python=3.9 -y
conda activate seg_env
conda install -c conda-forge dcm2niix -y
pip install TotalSegmentator pyradiomics SimpleITK pandas numpy xgboost scikit-learn

# Convert DICOM to NIfTI
echo "=== STEP 2: Converting DICOM to NIfTI ==="
python3 01_dcm2nii.py

# Run TotalSegmentator
echo "=== STEP 3: Running TotalSegmentator ==="
python3 02_run_totalseg.py

# Extract PyRadiomics
echo "=== STEP 4: Extracting PyRadiomics ==="
python3 03_extract_radiomics.py

echo "=== OVERNIGHT JOB COMPLETE ==="
