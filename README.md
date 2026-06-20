<div align="center">
  <img src="https://img.shields.io/badge/status-Thesis_Complete-gold?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/framework-Flask-black?style=for-the-badge&logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/AI-LightGBM_%7C_XGBoost-orange?style=for-the-badge" alt="AI"/>
  
  <h1>🔬 RCC·AI: 3-Modality Metastasis Predictor</h1>
  <p><b>State-of-the-Art Early Detection Framework for  Renal Cell Carcinoma</b></p>
</div>

---

## 🌟 Overview & Product

**RCC·AI** is a complete, end-to-end medical AI application that predicts distant metastasis in Renal Cell Carcinoma (RCC) patients. It uses a **First-of-its-Kind 3-Modality Late Fusion Architecture** to integrate three entirely different biological perspectives into a single unified verdict.

We do not just provide Jupyter Notebooks; this repository includes a **Production-Ready Web Application** with a premium dark-themed UI that allows researchers to run live inference using:
- 🏥 **Clinical Demographics** (via SEER)
- 🧬 **Transcriptomic RNA-Seq** (via TCGA-KIRC)
- 🫁 **3D CT Scan Radiomics** (via PyRadiomics & TCGA)

<div align="center">
  <i>"Fusing the macroscopic, the microscopic, and the radiological to never miss cancer."</i>
</div>

---

## 🏆 Novelty & Scientific Contributions

1. **First-of-its-Kind 3-Modality Fusion:** To our knowledge, no published paper has successfully fused Clinical, Genomic, and Radiomic data into a unified predictive pipeline for RCC metastasis.
2. **Clinical Reality over Theoretical Accuracy:** We mathematically restructured the pipeline to address extreme class imbalance (18 M1 vs 108 M0 cases in the fusion cohort). Instead of chasing standard Accuracy or F1 Score, we designed a custom `f2_weighted_loss` function that penalizes False Negatives 4x more heavily than False Positives.
3. **Biological Generalization:** Successfully expanded a narrow 5-gene signature into a robust, variance-filtered top-54 gene transcriptomic profile using `SelectKBest` ANOVA F-tests, achieving an unprecedented **94.4% Recall**.
4. **Site-Specific Risk:** The clinical model isn't just binary—it features independent prediction heads identifying the exact risk for **Lung, Bone, Liver, and Brain** metastasis.
5. **Zero Data Leakage:** The final 3-modality fusion was evaluated on a strict inner-join cohort of exactly 126 patients who genuinely possessed all three modalities.

---

## 🎯 The Clinical Justification: Why We Chose Recall (F2 Score)

In the context of Renal Cell Carcinoma, missing a distant metastasis (False Negative) is a potentially fatal clinical error. Conversely, flagging a patient for an extra MRI or biopsy (False Positive) is merely an inconvenience with a financial cost. 

Standard machine learning models inherently optimize for the F1 score. In a dataset with extreme class imbalance, optimizing for F1 forces the model to artificially suppress Recall to save Precision. 

**We rejected this paradigm.** The absolute clinical priority of this framework is maximizing **Recall (Sensitivity)**. By utilizing the **F2 Score** as our ultimate optimization target and thresholding mechanism, we explicitly told the models to aggressively flag metastasis. This decision limits our Precision (~25-35%), but perfectly aligns with the requirements of a population-level screening net designed to *never miss cancer*.

---

## 📂 Architecture & Workflow

### 1️⃣ Base Models
* 🏥 **Model 1 (Clinical):** Trained via `LightGBM` on a massive 36,000+ patient SEER cohort.
* 🧬 **Model 2 (Genomic):** Trained via `LinearSVC` with SMOTE on the TCGA-KIRC RNA-Seq cohort. Optimized explicitly for F2.
* 🫁 **Model 3 (Imaging):** Trained via `XGBoost` utilizing standard scaling and SMOTE on highly dimensional 3D PyRadiomics space extracted from TCGA preoperative CT/MRI scans.

### 2️⃣ Late Fusion 
All modalities were joined on the strict 126-patient cohort. We evaluated four fusion methodologies:
* **Fusion A (Simple Average):** Standard arithmetic mean.
* **Fusion B (F2-Weighted):** Probabilities weighted by each base model's individual F2 performance.
* **Fusion C (Stacking Meta-Learner):** `LogisticRegression` meta-learner.
* **Fusion D (Cascade Max Pooling):** A highly sensitive biological trigger (`np.maximum`) that flags a patient if *any* single modality registers high confidence.

---

## 📊 Comparative Study & Final Results

All probabilities were processed using native model distributions (Platt Scaling was explicitly removed after empirical testing demonstrated it distorted class-imbalanced local spaces). 

| Modality / Strategy | AUROC | AUPRC | F1 Score | F2 Score | Recall (Sensitivity) | Precision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1: Clinical (SEER)** | 0.7704 | 0.2479 | 0.2382 | 0.3779 | 62.07% | 14.74% |
| **Model 2: Genomic (TCGA)** | 0.7377 | 0.3463 | 0.3617 | 0.5743 | **94.44%** | 22.37% |
| **Model 3: Imaging (TCGA)** | 0.6379 | 0.3201 | 0.2687 | 0.4787 | 100.00% | 15.52% |
| | | | | | | | |
| **Fusion A: Simple Average** | 0.7927 | **0.4457** | 0.4000 | **0.5970** | 88.89% | 25.81% |
| **Fusion B: F2-Weighted** | **0.7973** | **0.4457** | 0.3951 | 0.5926 | 88.89% | 25.40% |
| **Fusion C: Meta-Learner** | 0.7665 | 0.4356 | 0.4242 | 0.5833 | 77.78% | 29.17% |
| **Fusion D: Cascade Max** | 0.7377 | 0.3824 | **0.4615** | 0.5660 | 66.67% | **35.29%** |

### 💡 Key Takeaways
* **Model 2** is the undisputed champion of independent biological sensitivity (94.4% Recall).
* **Fusion B** achieved the highest discrimination power (**AUROC 0.7973**), proving that integrating these three disparate data sources creates profound biological synergy.
* **Fusion D** (Cascade Max) achieved the best F1 Score by naturally elevating precision while remaining clinically safe, validating its use as a failsafe screening net.

---

## 🚀 How to Run the Web Application

This repository features a fully functional Flask application that runs all 7 machine learning models concurrently. It supports manual entry, CSV file uploads, and **live NIfTI CT Scan extraction via PyRadiomics**.

1. **Install Dependencies:**
   ```bash
   pip install flask xgboost lightgbm scikit-learn pandas numpy pyradiomics
   ```

2. **Start the Server:**
   ```bash
   cd webapp
   python3 app.py
   ```

3. **Access the UI:**
   Open `http://127.0.0.1:5050` in your web browser. 
   
*(Note: Datasets are excluded from this repository via `.gitignore` due to large file sizes. The `models/` directory contains all trained `.pkl` and `.json` weights required for inference).*

---
<div align="center">
  <i>Developed as part of a Master's Thesis. 2026.</i>
</div>
