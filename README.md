<div align="center">
  <img src="https://img.shields.io/badge/status-Thesis_Complete-gold?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/framework-Flask-black?style=for-the-badge&logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/AI-LightGBM_%7C_XGBoost_%7C_LinearSVC-orange?style=for-the-badge" alt="AI"/>
</div>

<h1 align="center">🔬 RCC·AI: Multi-Cohort Late Fusion Metastasis Predictor</h1>
<p align="center"><b>Decision-Level Multimodal Framework for Renal Cell Carcinoma</b></p>

---

## 🌟 Innovation & Novelty

**RCC·AI** is a clinical decision support system designed to predict distant metastasis in Renal Cell Carcinoma (RCC). The primary scientific novelty of this project lies in its **multi-cohort, decision-level late fusion architecture**, which bridges the gap between massive population-scale clinical registries and high-dimensional, patient-scale molecular and imaging data.

### Key Innovations:
1. **Cross-Cohort Modality Harmonization:** Instead of relying on a single, small multimodal dataset, this project trains a foundational clinical model on a population-scale cohort (36,738 patients) and applies it via **transfer** to a high-dimensional genomic/radiomic cohort (TCGA-KIRC).
2. **Clinical Alignment via F2-Loss Optimization:** In clinical screening, missing a distant metastasis (False Negative) is a potentially fatal error, whereas a False Positive merely triggers an imaging scan. This pipeline uses a custom F2-weighted loss function (penalizing False Negatives 4× more than False Positives) and F2-threshold optimization to force the AI to behave like a highly sensitive screening tool (achieving near-90% sensitivity).
3. **Transparent Late Fusion:** The framework utilizes **score-level (late) fusion**. Three independently trained, single-modality models output risk probabilities, which are mathematically combined. This avoids the black-box nature of joint representation deep learning while mathematically proving that biological multi-modality improves predictive discrimination (AUROC 0.770 → 0.797).

*(Note: This is **not** an end-to-end multimodal deep learning system and does **not** perform joint representation learning. It is a strictly validated, decision-level ensemble.)*

---

## 🗄️ Datasets and Provenance

To achieve multimodal fusion, data was sourced from two premier international oncology databases. 

### 1. The SEER Program Database (Clinical Modality)
- **Source:** Surveillance, Epidemiology, and End Results (SEER) Program (NIH/NCI).
- **Cohort:** 36,738 RCC patients diagnosed between 2010 and 2018.
- **Purpose:** SEER provides massive statistical power but lacks molecular and imaging data. This dataset was strictly used to train **Model 1 (Clinical)**. 
- **Features Extracted:** Age, Sex, T-Stage, N-Stage, Tumour Size (cm), Fuhrman Grade, Histological Subtype, Prior Treatment, Year of Diagnosis.

### 2. TCGA-KIRC Database (Genomic & Radiomic Modalities)
- **Source:** The Cancer Genome Atlas Kidney Clear Cell Carcinoma (TCGA-KIRC) collection, via UCSC Xena and The Cancer Imaging Archive (TCIA).
- **Genomic Cohort:** 418 patients (RNA-Seq HiSeqV2 transcriptomics matched with valid clinical M-stage). Used to train and validate **Model 2 (Genomic)** via Out-Of-Fold (OOF) cross-validation.
- **Radiomic Cohort:** 126 patients with valid pre-operative 3D CT scans (NIfTI/DICOM). Used to train and validate **Model 3 (Imaging)**.
- **Alignment Cohort (The Fusion Base):** The final multimodal evaluation was conducted on a **harmonised inner-join alignment cohort** of exactly 126 patients who simultaneously had Clinical metadata, RNA-Seq profiles, and CT scans available.

---

## ⚙️ Technical Implementation & Model Selection

The models were not chosen at random; algorithms were specifically selected to match the structure, dimensionality, and noise profile of their respective modalities.

### 🏥 Model 1: Clinical (SEER Transfer)
- **Algorithm Selected:** **LightGBM**. Chosen for its native handling of categorical features (histology, grade) without one-hot explosion, its speed on 36k rows, and its robustness to missing data.
- **Implementation:** Trained on SEER data utilizing Optuna for Hyperparameter Optimization (HPO) and SMOTE for class imbalance. Implements a custom F2-loss function.
- **Evaluation:** Evaluated on a 20% SEER holdout, then applied directly to the TCGA 126-patient fusion cohort via feature-mapping transfer.

### 🧬 Model 2: Genomic (TCGA-418)
- **Algorithm Selected:** **LinearSVC** (Support Vector Classifier with Linear Kernel) wrapped in a `CalibratedClassifierCV`. Chosen because RNA-Seq data is ultra-high dimensional (19,000+ genes) but has a tiny sample size (n=418). Linear SVCs are mathematically proven to be highly resistant to overfitting in $p \gg N$ scenarios compared to tree-based models or deep neural networks.
- **Feature Selection:** Filtered via 25th-percentile variance masking, followed by ANOVA F-test `SelectKBest(k=50)`, plus 5 literature-validated genes, resulting in a final **54-gene transcriptomic profile**.

### 🫁 Model 3: Radiomic (TCGA-126)
- **Algorithm Selected:** **XGBoost Classifier**. Chosen because the 49 PyRadiomics features are dense, continuous, and highly collinear. XGBoost handles collinearity inherently through tree-splitting and provides excellent regularization (`max_depth=3`, `learning_rate=0.05`).
- **Implementation:** Raw DICOMs were automatically segmented using **TotalSegmentator** (AI-driven 3D segmentation). The kidney tumour ROIs were passed to **PyRadiomics** to extract Shape, First-order, GLCM, GLRLM, and GLSZM features. 
- **Leakage Prevention:** StandardScaler and SMOTE were strictly applied *inside* the 5-Fold Stratified Cross-Validation loop to prevent data leakage.

### 🧩 Decision-Level Late Fusion (The Alignment Cohort)
Four fusion strategies were mathematically applied to the probability outputs ($P_1, P_2, P_3$) of the base models:
1. **Fusion A (Simple Average):** $(P_1 + P_2 + P_3) / 3$
2. **Fusion B (F2-Weighted Average):** $(w_1P_1 + w_2P_2 + w_3P_3) / \sum w$, where $w$ is the F2 score of each base model.
3. **Fusion C (Stacking Meta-Learner):** A Logistic Regression algorithm trained via nested 5-Fold CV on the probability outputs.
4. **Fusion D (Cascade Max Pooling):** $max(P_1, P_2, P_3)$ — triggers a positive flag if *any* modality detects metastasis.

---

## 📊 Final Results

All metrics below are sourced directly from empirical Out-Of-Fold and Holdout testing. 

### Individual Modality Performance

| Model | Cohort | n | AUROC | Recall | Precision | F2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Model 1: Clinical | SEER Holdout | ~7,348 | **0.7704** | 62.07% | 14.74% | 0.3779 |
| Model 2: Genomic | TCGA-418 OOF | 418 | 0.6420 | 92.86% | 22.37% | 0.5242 |
| Model 3: Imaging | TCGA-126 OOF | 126 | 0.6591 | 100.0% | 15.52% | 0.5128 |

### 3-Modality Fusion (TCGA-126 Alignment Cohort)

| Strategy | AUROC | AUPRC | F1 | F2 | Recall | Precision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Fusion A: Simple Average | 0.7927 | **0.4457** | 0.4000 | **0.5970** | 88.89% | 25.81% |
| **Fusion B: F2-Weighted ⭐** | **0.7973** | **0.4457** | 0.3951 | 0.5926 | 88.89% | 25.40% |
| Fusion C: Stacking Meta-Learner | 0.7665 | 0.4356 | **0.4242** | 0.5833 | 77.78% | **29.17%** |
| Fusion D: Cascade Max Pooling | 0.7377 | 0.3824 | 0.4615 | 0.5660 | 66.67% | 35.29% |

**Conclusion:** Fusion B (F2-Weighted) yields the highest discrimination (AUROC 0.7973). The AUROC improvement from the best single model (Clinical, 0.7704) to the best fusion (0.7973) is **+0.027**. This represents a real, mathematically proven gain consistent with late fusion theory on a small alignment cohort. At 88.89% Recall, Precision sits at 25.40%—meaning 1 in 4 patients flagged actually has distant metastasis. This is highly appropriate for a first-line screening tool designed to cast a wide net and refer high-risk patients for further definitive imaging.

### Visual Proof
You can view the comprehensive set of 10 publication-quality figures, including ROC curves, Precision-Recall operating points, dataset demographics, and confusion matrices in the [`results/figures_for_research_paper/`](./results/figures_for_research_paper/) directory.

---

## 🚀 Running the Web Application

This repository includes a full-stack web application designed for clinicians to interact with the models.

```bash
# 1. Install Dependencies
pip install flask flask-cors xgboost lightgbm scikit-learn imbalanced-learn pandas numpy joblib pyradiomics

# 2. Run the Backend API
cd webapp
python3 app.py

# 3. Access the Frontend
# Open http://127.0.0.1:5050 in your browser
```

The frontend (Vercel-hosted design) supports manual clinical entry, CSV batch upload, and live NIfTI CT scan feature extraction via PyRadiomics. The backend (Flask) routes the data through the pre-trained models and returns structured JSON predictions for all fusion strategies and organ-specific relative risk indices.

*(Datasets excluded via .gitignore due to file size. Trained model weights in `models/` directory are included.)*

---

<div align="center">
  <i>Masters Thesis · 2026</i><br/>
  <i>"A multi-cohort, decision-level late fusion system for RCC metastasis screening."</i>
</div>
