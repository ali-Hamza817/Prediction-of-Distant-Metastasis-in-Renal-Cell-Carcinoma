<div align="center">
  <h1>🔬 RCC·AI</h1>
  <h3>A Decision-Level Fusion Framework for Predicting Distant Metastasis in Renal Cell Carcinoma Using Clinical, Transcriptomic, and Radiomic Biomarkers</h3>

  <p align="center">
    <img src="https://img.shields.io/badge/Status-Masters_Thesis_Complete-2ea44f?style=for-the-badge&logo=googlescholar&logoColor=white" alt="Status"/>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/Flask-Web_Backend-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
    <img src="https://img.shields.io/badge/Vercel-Frontend-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel"/>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Clinical_AI-LightGBM-6366f1?style=for-the-badge" alt="LightGBM"/>
    <img src="https://img.shields.io/badge/Genomic_AI-LinearSVC-0891b2?style=for-the-badge" alt="SVC"/>
    <img src="https://img.shields.io/badge/Radiomic_AI-XGBoost-059669?style=for-the-badge" alt="XGBoost"/>
  </p>
</div>

---

## 🛑 1. Problem Statement

Renal Cell Carcinoma (RCC) is the most common type of kidney cancer. Approximately **25–30% of patients present with distant metastasis at initial diagnosis**, and a significant portion of localized cases eventually metastasize. Distant metastasis drastically reduces the 5-year survival rate to below 15%. 

Currently, metastasis risk is evaluated using clinical nomograms (like SSIGN or UISS), which rely solely on low-dimensional clinical factors (tumour size, grade, necrosis). They completely ignore the rich molecular (transcriptomic) and spatial (radiomic) biology of the tumour, leading to a high rate of missed early metastases (False Negatives). Missing a metastasis in RCC is a potentially fatal clinical error.

## 💡 2. The Solution: RCC·AI

**RCC·AI** is a comprehensive clinical decision support system designed to predict distant metastasis in RCC by leveraging **decision-level late fusion** across three disparate biological scales:
1. **Population-Scale Clinical Factors** (Demographics, Staging, Pathology)
2. **Molecular-Scale Genomics** (RNA-Seq Transcriptomics)
3. **Macro-Scale Radiomics** (3D CT Scan Textures and Morphologies)

Instead of relying on a single modality, RCC·AI trains independent, highly specialized AI models for each modality. It then fuses their probability outputs to create a highly sensitive, multimodal safety net that ensures high-risk patients are not missed.

## 🎯 3. Project Objectives

- **Primary Objective:** Develop a multi-cohort, multimodal predictive framework that outperforms single-modality clinical models in predicting RCC distant metastasis.
- **Secondary Objective 1:** Mathematically align the model's loss function to clinical reality, explicitly optimizing for extremely high Recall (Sensitivity) to minimize False Negatives.
- **Secondary Objective 2:** Deploy the architecture into an accessible, physician-facing web application that returns both an overall metastasis probability and organ-specific relative risk indices (Lung, Bone, Liver, Brain).

## 🌟 4. Innovation & Novelty

1. **Cross-Cohort Modality Harmonization:** Instead of training a small multimodal model from scratch, this project trains a foundational clinical model on a massive population registry (36,738 SEER patients) and applies it via **transfer learning** to a high-dimensional, patient-scale cohort (TCGA-KIRC).
2. **Clinical Alignment via F2-Loss Optimization:** This pipeline utilizes a custom F2-weighted loss function (penalizing False Negatives 4× more than False Positives). This forces the AI to behave like a highly sensitive, first-line screening tool, achieving near-90% sensitivity.
3. **Transparent Late Fusion:** The framework utilizes **score-level (late) fusion**. This avoids the black-box opacity of end-to-end deep representation learning. By combining independent probabilities mathematically, the model remains interpretable while still capturing the biological synergy of multiple modalities.

---

## 🔄 5. Transfer Learning & Its Proven Benefit

A major biological and data science hurdle in oncology is that population-scale registries (like SEER) have massive patient numbers but no molecular data, while molecular databases (like TCGA) are incredibly rich but have tiny sample sizes. Training a multimodal model from scratch on 126 patients would instantly overfit.

**How we solved it via Domain Transfer:**
1. **Pre-training:** Model 1 (Clinical) is trained on 36,738 SEER patients, learning incredibly robust, population-level patterns about how Age, Tumour Size, T-Stage, and Grade relate to metastasis.
2. **Feature Harmonization:** The clinical metadata from the TCGA patients is mathematically mapped to match the input format expected by the SEER model.
3. **Zero-Shot Inference (The Transfer):** We take the pre-trained SEER model and run it directly on the 126 TCGA patients **without retraining or fine-tuning**. 

**The Benefit:** 
By transferring the SEER model to TCGA, we effectively inject the statistical weight and confidence of 36,000+ patients into a tiny 126-patient multimodal cohort. This proves that we do not need a single, massive dataset containing all three modalities (which is exceedingly rare); instead, we can train highly specialized models on disjointed datasets and mathematically fuse them via transfer learning.

---

## 🗄️ 6. Datasets & Provenance

To achieve multimodal fusion without compromising statistical power, data was sourced from two premier international oncology databases.

### The SEER Program Database (Clinical Modality)
- **Source:** Surveillance, Epidemiology, and End Results (SEER) Program.
- **Cohort:** 36,738 RCC patients (diagnosed 2010–2018).
- **Purpose:** SEER provides massive statistical power but lacks molecular/imaging data. Used strictly to train **Model 1 (Clinical)**.

### TCGA-KIRC Database (Genomic & Radiomic Modalities)
- **Source:** The Cancer Genome Atlas Kidney Clear Cell Carcinoma (TCGA-KIRC).
- **Genomic Cohort:** 418 patients (RNA-Seq HiSeqV2 transcriptomics matched with valid clinical M-stage). Used to train **Model 2 (Genomic)** via Out-Of-Fold CV.
- **Radiomic Cohort:** 126 patients with valid pre-operative 3D CT scans. Segmented via TotalSegmentator and used to train **Model 3 (Imaging)**.
- **The Alignment Cohort:** The final 3-modality fusion evaluation was conducted strictly on a **harmonised inner-join alignment cohort** of the 126 patients who simultaneously possessed complete Clinical, Genomic, and Radiomic data.

---

## ⚙️ 7. Technical Implementation Details

Algorithms were specifically selected to match the structure, dimensionality, and noise profile of their respective modalities.

### 🏥 Model 1: Clinical (SEER Transfer)
- **Algorithm:** **LightGBM**. Chosen for its native handling of categorical features (histology, Fuhrman grade), its speed on 36k+ rows, and its robustness to missing data.
- **Implementation:** Optuna Hyperparameter Optimization + SMOTE. Implements the custom F2-loss function. Evaluated on a 20% SEER holdout, then transferred to TCGA.
- **Organ-Specific Sub-Models:** 4 additional LightGBM models trained specifically to predict metastasis sites (Lung, Bone, Liver, Brain) as relative risk indices.

### 🧬 Model 2: Genomic (TCGA-418)
- **Algorithm:** **LinearSVC** (Support Vector Classifier, Linear Kernel) wrapped in `CalibratedClassifierCV`. Chosen because RNA-Seq is ultra-high dimensional (19k+ genes) with a tiny sample size ($p \gg N$). SVCs mathematically resist overfitting in this exact scenario.
- **Feature Selection:** 25th-percentile variance masking → ANOVA F-test `SelectKBest(k=50)` → 5 literature-validated genes. Resulting in a final **54-gene transcriptomic profile**.

### 🫁 Model 3: Radiomic (TCGA-126)
- **Algorithm:** **XGBoost Classifier**. Chosen because the 49 PyRadiomics features are dense, continuous, and highly collinear. Tree-based splitting inherently manages collinearity with high regularization (`max_depth=3`).
- **Implementation:** Pre-operative DICOMs automatically segmented via **TotalSegmentator**. PyRadiomics used to extract Shape, First-order, GLCM, GLRLM, and GLSZM features. 

### 🧩 Decision-Level Late Fusion
Four fusion strategies were mathematically applied to the probability outputs ($P_1, P_2, P_3$):
1. **Fusion A (Simple Average):** Arithmetic mean.
2. **Fusion B (F2-Weighted Average):** Weighted mean, where weights are the F2 scores of the base models.
3. **Fusion C (Stacking Meta-Learner):** Logistic Regression trained via nested 5-Fold CV on the probabilities.
4. **Fusion D (Cascade Max Pooling):** $max(P_1, P_2, P_3)$ — triggers a positive flag if *any* modality detects high risk.

---

## 📊 8. Final Results

All metrics below are sourced directly from empirical Out-Of-Fold and Holdout testing. 

### Base Modality Performance

| Model | Cohort | n | AUROC | Recall | Precision | F2 Score |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Model 1: Clinical | SEER Holdout | ~7,348 | **0.7704** | 62.07% | 14.74% | 0.3779 |
| Model 2: Genomic | TCGA-418 OOF | 418 | 0.6420 | 92.86% | 22.37% | 0.5242 |
| Model 3: Imaging | TCGA-126 OOF | 126 | 0.6591 | 100.0% | 15.52% | 0.5128 |

### 3-Modality Fusion (126-Patient Alignment Cohort)

| Strategy | AUROC | AUPRC | Recall | Precision | F2 Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| Fusion A: Simple Average | 0.7927 | **0.4457** | 88.89% | 25.81% | **0.5970** |
| **Fusion B: F2-Weighted ⭐** | **0.7973** | **0.4457** | 88.89% | 25.40% | 0.5926 |
| Fusion C: Stacking Meta-Learner | 0.7665 | 0.4356 | 77.78% | **29.17%** | 0.5833 |
| Fusion D: Cascade Max Pooling | 0.7377 | 0.3824 | 66.67% | 35.29% | 0.5660 |

**Conclusion:** Fusion B (F2-Weighted) yields the highest discrimination (AUROC 0.7973). The AUROC improvement from the best single model (0.770) to the best fusion (0.797) proves the biological synergy of late multimodal fusion. At 88.89% Recall, Precision sits at 25.40%—highly appropriate for a first-line screening tool designed to cast a wide net and refer high-risk patients for definitive imaging.

> 🖼️ **Visual Proof:** View the comprehensive set of 10 publication-quality figures, including ROC curves, Precision-Recall points, and dataset demographics in the [`results/figures_for_research_paper/`](./results/figures_for_research_paper/) directory.

---

## ⚠️ 9. Scientific Limitations

1. **Alignment Cohort Selection Bias:** The 126-patient inner join is not a natural dataset. Patients with incomplete multimodal data (e.g., failed imaging segmentation, missing RNA-Seq) were excluded, which may introduce spectrum bias.
2. **Small Positive Class in Fusion:** Only 18 of the 126 patients had true metastasis (M1). All fusion metrics possess wide confidence intervals due to this severe class imbalance.
3. **Late Fusion Ceiling:** Score-level fusion is constrained by the weakest base model. Joint representation deep learning on a much larger cohort would be required to break the ~0.80 AUROC ceiling.

---

## 🚀 10. Running the Application

This repository includes a full-stack web application designed for clinician interaction.

```bash
# 1. Install Dependencies
pip install flask flask-cors xgboost lightgbm scikit-learn imbalanced-learn pandas numpy joblib pyradiomics

# 2. Run the Backend API
cd webapp
python3 app.py

# 3. Access the Frontend
# Open http://127.0.0.1:5050 in your browser
```

The frontend (Vercel-hosted design) supports manual clinical entry, CSV batch upload, and live NIfTI CT scan feature extraction via PyRadiomics. The backend (Flask) routes the data through the pre-trained `.pkl`/`.json` models and returns structured JSON predictions for all fusion strategies and organ-specific relative risk indices.

*(Note: Datasets are excluded via .gitignore due to size limitations. Trained model weights in the `models/` directory are included.)*
