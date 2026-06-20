import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 0: Title
cells.append(nbf.v4.new_markdown_cell("""# Model 2: Genomic (TCGA Dataset)
This notebook trains a Linear SVC model on a dynamically selected variance-filtered gene signature to predict distant metastasis. It utilizes the full TCGA cohort and generates Out-of-Fold (OOF) predictions."""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, 
                             precision_score, recall_score, f1_score, fbeta_score)
from imblearn.over_sampling import SMOTE
import warnings
import os
import joblib

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
"""))

# Cell 2: Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Loading & Alignment"""))
cells.append(nbf.v4.new_code_cell("""# Load clinical data
clin_path = '../datasets/dataset_2/KIRC_clinicalMatrix.tsv'
clin_df = pd.read_csv(clin_path, sep='\t')

# Filter for usable pathologic_M patients
usable_clin = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
usable_clin['metastasis'] = (usable_clin['ajcc_m'] == 'M1').astype(int)

# Set patient ID as index
usable_clin.set_index('submitter_id', inplace=True)
usable_clin.index = usable_clin.index.str[:12]

# Load RNA-seq data (Full 20,000+ genes)
rna_path = '../datasets/dataset_2/HiSeqV2.gz'
rna_df = pd.read_csv(rna_path, sep='\t', index_col=0, compression='gzip').T
rna_df.index = rna_df.index.str[:12]

# Inner Join exactly on the 418 valid patients
df = usable_clin[['metastasis']].join(rna_df, how='inner')
df = df[~df.index.duplicated(keep='first')]
df = df.dropna()

print(f"Full TCGA Genomic Cohort: {df.shape[0]} patients")
print(f"Class distribution:\\n{df['metastasis'].value_counts()}")

y = df['metastasis']
X_raw = df.drop(columns=['metastasis'])
"""))

# Cell 3: Feature Engineering (Fix 1)
cells.append(nbf.v4.new_markdown_cell("""## 2. Feature Selection"""))
cells.append(nbf.v4.new_code_cell("""# Step 1: Remove near-zero variance genes
var = X_raw.var()
X_filtered = X_raw.loc[:, var > var.quantile(0.25)]
print(f"Genes after 25% variance filter: {X_filtered.shape[1]}")

# Step 2: Select top 50 genes associated with metastasis
selector = SelectKBest(f_classif, k=50)
selector.fit(X_filtered, y)
selected_genes = X_filtered.columns[selector.get_support()]
print(f"Top 50 selected genes: {list(selected_genes[:10])}...")

# Step 3: Always include the Paper 2 signature genes
paper2_genes = ['FKBP15', 'SLC31A1', 'CPT2', 'PATJ', 'CALR']
# Only include paper2_genes if they exist in X_raw
paper2_genes = [g for g in paper2_genes if g in X_raw.columns]
all_genes = list(set(selected_genes) | set(paper2_genes))
X = X_raw[all_genes]

print(f"Final genomic feature set: {len(all_genes)} genes")
"""))

# Cell 4: Evaluation Strategy
cells.append(nbf.v4.new_markdown_cell("""## 3. Evaluation Strategy & Metrics"""))
cells.append(nbf.v4.new_code_cell("""def evaluate_model(y_true, y_prob, threshold=0.5):
    auroc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_curve, precision_curve)
    
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    
    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'F2': f2
    }
"""))

# Cell 5: Out-of-Fold Predictions (Fix 2 & Fix 3)
cells.append(nbf.v4.new_markdown_cell("""## 4. Train Linear SVC & Generate Out-Of-Fold (OOF) Predictions"""))
cells.append(nbf.v4.new_code_cell("""# 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Linear SVC wrapped in CalibratedClassifierCV to get probabilities
svm = LinearSVC(C=0.01, class_weight='balanced', max_iter=5000)
model2 = CalibratedClassifierCV(svm, cv=5)

# Initialize predictions array
p2_full = np.zeros(len(y))

# Manual loop for preprocessing within CV
for train_idx, test_idx in skf.split(X, y):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)
    
    # SMOTE
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train_fold)
    
    # Fit & Predict
    model2.fit(X_train_sm, y_train_sm)
    p2_full[test_idx] = model2.predict_proba(X_test_scaled)[:, 1]

# Overall Evaluation
print("\\n--- Model 2 Final OOF Results (Full TCGA Cohort n=418) ---")
results = evaluate_model(y, p2_full, threshold=0.5)
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Train Final Model on all data
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
X_sm, y_sm = SMOTE(sampling_strategy=0.5, random_state=42).fit_resample(X_scaled, y)
model2.fit(X_sm, y_sm)

# Save Model and Scaler
os.makedirs('../models/dataset_2', exist_ok=True)
joblib.dump(model2, '../models/dataset_2/Model2_Genomic_TCGA.pkl')
joblib.dump(final_scaler, '../models/dataset_2/Model2_Scaler.pkl')
joblib.dump(list(X.columns), '../models/dataset_2/Model2_Features.pkl')

# Save OOF Predictions for Late Fusion
oof_df = pd.DataFrame({'submitter_id': y.index, 'P2': p2_full, 'metastasis': y.values})
oof_df.to_csv('../results/Model2_OOF_Predictions.csv', index=False)
print("\\nModel 2 saved and OOF Predictions exported for Fusion.")
"""))

nb.cells = cells
nbf.write(nb, '/home/administrator/Desktop/RCC/notebooks/02_Model2_Genomic_TCGA.ipynb')
print("Notebook 02_Model2_Genomic_TCGA.ipynb generator created successfully.")
