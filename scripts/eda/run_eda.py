import pandas as pd
import numpy as np
import os
import pydicom
import random
import warnings

warnings.filterwarnings('ignore')

def eda_dataset_1():
    print("Running EDA for Dataset 1...")
    path = 'datasets/dataset_1/seer_rcc_2010_2018_clean.csv'
    df = pd.read_csv(path)
    
    report = f"""# Dataset 1 (Clinical SEER) - EDA Report

## Overview
- **File:** `seer_rcc_2010_2018_clean.csv`
- **Rows (Patients):** {df.shape[0]:,}
- **Columns:** {df.shape[1]}

## Missing Values
```text
{df.isnull().sum().to_string()}
```

## Class Distribution (Main Target: metastasis)
- M0 (No Metastasis): {(df['metastasis']==0).sum():,}
- M1 (Metastasis): {(df['metastasis']==1).sum():,}
- **M1 Prevalence:** {(df['metastasis'].mean()*100):.2f}%

## Site-Specific Metastasis Counts
- Lung: {(df.get('lung_met', pd.Series(dtype=int))==1).sum():,}
- Bone: {(df.get('bone_met', pd.Series(dtype=int))==1).sum():,}
- Liver: {(df.get('liver_met', pd.Series(dtype=int))==1).sum():,}
- Brain: {(df.get('brain_met', pd.Series(dtype=int))==1).sum():,}

## Continuous Features Summary
```text
{df[['age', 'tumor_size_cm', 'survival_months']].describe().to_string()}
```
"""
    with open('datasets/dataset_1/EDA_Report.md', 'w') as f:
        f.write(report)
        
def eda_dataset_2():
    print("Running EDA for Dataset 2...")
    clin_path = 'datasets/dataset_2/KIRC_clinicalMatrix.tsv'
    rna_path = 'datasets/dataset_2/HiSeqV2.gz'
    
    clin_df = pd.read_csv(clin_path, sep='\\t')
    
    # We want to know usable patients based on pathologic_M
    # M stage column is 'pathologic_M'
    if 'pathologic_M' in clin_df.columns:
        m_counts = clin_df['pathologic_M'].value_counts()
        m_str = m_counts.to_string()
        
        usable_patients = clin_df[clin_df['pathologic_M'].isin(['M0', 'M1'])]
    else:
        m_str = "No 'pathologic_M' column found!"
        usable_patients = pd.DataFrame()
        
    try:
        rna_df = pd.read_csv(rna_path, sep='\\t', index_col=0)
        # Transpose so rows are patients, cols are genes
        rna_df = rna_df.T
        
        genes = ['FKBP15', 'SLC31A1', 'CPT2', 'PATJ', 'CALR']
        available_genes = [g for g in genes if g in rna_df.columns]
        
        gene_summary = rna_df[available_genes].describe().to_string() if available_genes else "Genes not found."
    except Exception as e:
        gene_summary = f"Error loading RNA-seq: {e}"

    report = f"""# Dataset 2 (Genomics / TCGA) - EDA Report

## Overview
- **Clinical File:** `KIRC_clinicalMatrix.tsv` ({clin_df.shape[0]} patients)
- **RNA-seq File:** `HiSeqV2.gz`

## Clinical Labels (pathologic_M)
```text
{m_str}
```
**Usable M0/M1 Patients:** {usable_patients.shape[0]}

## 5-Gene Expression Summary
The target genes identified by the Naeini et al. 2026 baseline.
```text
{gene_summary}
```
"""
    with open('datasets/dataset_2/EDA_Report.md', 'w') as f:
        f.write(report)
        
def eda_dataset_3():
    print("Running EDA for Dataset 3...")
    base_dir = 'datasets/dataset_3_TCGA-KIRC'
    
    # Find all DICOM files
    dicom_files = []
    patient_dirs = [d for d in os.listdir(base_dir) if d.startswith('TCGA')]
    
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.dcm'):
                dicom_files.append(os.path.join(root, f))
                
    total_patients = len(patient_dirs)
    total_dicoms = len(dicom_files)
    
    # Sample 50 dicom files to extract metadata
    sample_size = min(50, total_dicoms)
    sampled = random.sample(dicom_files, sample_size)
    
    modalities = set()
    slice_thicknesses = set()
    body_parts = set()
    
    for dcm_path in sampled:
        try:
            d = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            if 'Modality' in d: modalities.add(d.Modality)
            if 'SliceThickness' in d: slice_thicknesses.add(str(d.SliceThickness))
            if 'BodyPartExamined' in d: body_parts.add(d.BodyPartExamined)
        except:
            pass

    report = f"""# Dataset 3 (Imaging / TCGA-KIRC) - EDA Report

## Overview
- **Total Patients (Folders):** {total_patients}
- **Total DICOM Files:** {total_dicoms:,}

## Sampled DICOM Metadata
*(Based on a random sample of {sample_size} DICOM headers)*

- **Modalities Found:** {', '.join(modalities)}
- **Body Parts Examined:** {', '.join(body_parts) if body_parts else "Not Specified"}
- **Slice Thicknesses (mm):** {', '.join(slice_thicknesses)}

## Fusion Potential
These {total_patients} patients share IDs with Dataset 2. A Late Fusion framework is directly possible by taking the intersection of patient IDs!
"""
    with open('datasets/dataset_3_TCGA-KIRC/EDA_Report.md', 'w') as f:
        f.write(report)

if __name__ == '__main__':
    eda_dataset_1()
    eda_dataset_2()
    eda_dataset_3()
    print("EDA Generation Complete!")
