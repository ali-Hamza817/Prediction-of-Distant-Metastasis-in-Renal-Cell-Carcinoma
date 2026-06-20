#!/bin/bash
set -e

echo "Creating Python 3.9 environment for TotalSegmentator and PyRadiomics..."
source ~/miniconda3/etc/profile.d/conda.sh

# Create clean python 3.9 environment
conda create -n seg_env python=3.9 -y
conda activate seg_env

echo "Installing dcm2niix (via conda) and deep learning packages..."
conda install -c conda-forge dcm2niix -y

echo "Installing TotalSegmentator, PyRadiomics, and Data Science packages..."
pip install TotalSegmentator pyradiomics SimpleITK pandas numpy xgboost scikit-learn

echo "Environment seg_env setup successfully!"
