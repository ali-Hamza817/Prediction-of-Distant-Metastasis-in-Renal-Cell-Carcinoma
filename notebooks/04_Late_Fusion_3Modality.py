#!/usr/bin/env python
# coding: utf-8

# # Final 3-Modality Late Fusion
# This notebook fuses predictions from Model 1 (SEER Clinical), Model 2 (TCGA Genomic), and Model 3 (TCGA Imaging / Radiomics) using three strategies: Simple Average, Weighted Average, and Logistic Regression Stacking with Nested Cross-Validation.

# In[1]:


import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, 
                             precision_score, recall_score, f1_score, fbeta_score, roc_curve)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')


# In[2]:


def evaluate_model(y_true, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_curve, precision_curve)

    # We use optimal threshold for F2 / Recall
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_idx = [i for i, r in enumerate(recalls) if r >= 0.80]
    if len(valid_idx) > 0:
        optimal_idx = max(valid_idx, key=lambda i: precisions[i])
        threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    else:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {'AUROC': auroc, 'AUPRC': auprc, 'F2': f2, 'Recall': recall}


# ## 1. Data Alignment (Clinical + Genomic + Imaging on TCGA)

# In[3]:


# Load TCGA Clinical target
clin_path = '../datasets/dataset_2/KIRC_clinicalMatrix.tsv'
clin_df = pd.read_csv(clin_path, sep='\t')
clin_df = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
clin_df['metastasis'] = (clin_df['ajcc_m'] == 'M1').astype(int)
clin_df.set_index('submitter_id', inplace=True)
clin_df.index = clin_df.index.str[:12]

clin_features = pd.DataFrame(index=clin_df.index)
clin_features['age'] = clin_df['age_at_index'].fillna(clin_df['age_at_index'].mean())
clin_features['sex'] = clin_df['gender'].map({'male': 1, 'female': 0}).fillna(1)

def map_t_stage(t):
    if pd.isna(t): return 1
    if 'T1' in t: return 1
    if 'T2' in t: return 2
    if 'T3' in t: return 3
    if 'T4' in t: return 4
    return 1

clin_features['t_stage'] = clin_df['ajcc_t'].apply(map_t_stage)

def map_n_stage(n):
    if pd.isna(n): return 0
    if 'N0' in n: return 0
    if 'N1' in n: return 1
    if 'N2' in n: return 1
    return 0

clin_features['n_stage'] = clin_df['ajcc_n'].apply(map_n_stage)
clin_features['tumor_size_cm'] = 6.5
clin_features['grade'] = 2
clin_features['histology_enc'] = 0
clin_features['prior_tx'] = 0
clin_features['year_diagnosis'] = 2014

expected_clin_cols = ['age', 'sex', 't_stage', 'n_stage', 'tumor_size_cm', 'grade', 'histology_enc', 'prior_tx', 'year_diagnosis']
clin_features = clin_features[expected_clin_cols]

# Load Genomic Data (5 Genes)
gen_path = '../datasets/dataset_2/HiSeqV2.gz'
gen_df = pd.read_csv(gen_path, sep='\t', index_col=0).T
gen_df.index = gen_df.index.str[:12]
target_genes = ['FKBP15', 'SLC31A1', 'CPT2', 'PATJ', 'CALR']
available_genes = [g for g in target_genes if g in gen_df.columns]
gen_features = gen_df[available_genes]

# Load Imaging Data (Radiomics)
rad_path = '../datasets/dataset_3_radiomics.csv'
rad_df = pd.read_csv(rad_path)
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]

# Ensure unique indices before concat
clin_df = clin_df[~clin_df.index.duplicated(keep='first')]
clin_features = clin_features[~clin_features.index.duplicated(keep='first')]
gen_features = gen_features[~gen_features.index.duplicated(keep='first')]
rad_df = rad_df[~rad_df.index.duplicated(keep='first')]

# Inner Join across all 3 modalities
df = pd.concat([clin_df[['metastasis']], clin_features, gen_features, rad_df], axis=1, join='inner')

y_tcga = df['metastasis']
X_clin = df[expected_clin_cols]
X_gen = df[available_genes]

# Drop target column from radiomics dataframe features
rad_features_list = joblib.load('../models/dataset_3/Model3_Features.pkl')
X_rad = df[rad_features_list]

print(f"Final 3-Modality Fusion Cohort: {df.shape[0]} patients")
print(y_tcga.value_counts())


# ## 2. Generate Probabilities ($P_1$, $P_2$, $P_3$)

# In[4]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# P1: Model 1 (Clinical Transfer)
model1 = joblib.load('../models/dataset_1/Model1_Clinical_SEER.pkl')
P1_oof = model1.predict_proba(X_clin)[:, 1]

# P2: Model 2 (Genomic TCGA)
scaler_gen = StandardScaler()
X_gen_scaled = scaler_gen.fit_transform(X_gen)

model2_base = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42, max_iter=5000, class_weight='balanced')
P2_oof = cross_val_predict(model2_base, X_gen_scaled, y_tcga, cv=cv, method='predict_proba')[:, 1]

# P3: Model 3 (Imaging TCGA)
scaler_rad = StandardScaler()
X_rad_scaled = scaler_rad.fit_transform(X_rad)
X_rad_scaled = pd.DataFrame(X_rad_scaled, columns=X_rad.columns)

model3_base = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05, 
    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
)
P3_oof = cross_val_predict(model3_base, X_rad_scaled, y_tcga, cv=cv, method='predict_proba')[:, 1]

print("Successfully extracted P1 (Transfer), P2 (OOF), and P3 (OOF).")


# ## 3. 3-Modality Late Fusion Ablation Study

# In[5]:


metrics = {}
metrics['Model 1: Clinical (SEER)'] = evaluate_model(y_tcga, P1_oof)
metrics['Model 2: Genomic (TCGA)'] = evaluate_model(y_tcga, P2_oof)
metrics['Model 3: Imaging (TCGA)'] = evaluate_model(y_tcga, P3_oof)

# Strategy A: Simple Average
P_fusion_A = (P1_oof + P2_oof + P3_oof) / 3
metrics['Fusion A: Simple avg'] = evaluate_model(y_tcga, P_fusion_A)

# Strategy B: Weighted Average
w1 = metrics['Model 1: Clinical (SEER)']['AUROC']
w2 = metrics['Model 2: Genomic (TCGA)']['AUROC']
w3 = metrics['Model 3: Imaging (TCGA)']['AUROC']
P_fusion_B = (w1 * P1_oof + w2 * P2_oof + w3 * P3_oof) / (w1 + w2 + w3)
metrics['Fusion B: Weighted avg'] = evaluate_model(y_tcga, P_fusion_B)

# Strategy C: Stacking (Nested CV)
stack_features = np.column_stack([P1_oof, P2_oof, P3_oof])
meta_model = LogisticRegression(C=0.1, random_state=42)
P_fusion_C = cross_val_predict(meta_model, stack_features, y_tcga, cv=cv, method='predict_proba')[:, 1]
metrics['Fusion C: Stacking'] = evaluate_model(y_tcga, P_fusion_C)

# Compile Results
results_df = pd.DataFrame(metrics).T
print("\n--- Final Thesis Results: 3-Modality Late Fusion ---")
print(results_df.round(4))
results_df.to_csv('../results/Final_3Modality_Fusion_Results.csv')

