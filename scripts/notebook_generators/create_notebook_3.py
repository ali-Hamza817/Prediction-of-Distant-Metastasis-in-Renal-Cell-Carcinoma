import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Title
cells.append(nbf.v4.new_markdown_cell("""# Model 3: Imaging (TCGA-KIRC Radiomics)
This notebook trains an XGBoost model on Whole-Kidney PyRadiomics features to predict distant metastasis."""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, 
                             precision_score, recall_score, f1_score, fbeta_score)
from imblearn.over_sampling import SMOTE
from lifelines.utils import concordance_index
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
clin_df = pd.read_csv(clin_path, sep='\\t')

# Filter for usable pathologic_M patients
usable_clin = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
usable_clin['metastasis'] = (usable_clin['ajcc_m'] == 'M1').astype(int)

# Set patient ID as index
usable_clin.set_index('submitter_id', inplace=True)
usable_clin.index = usable_clin.index.str[:12]

# Load Radiomics data
rad_path = '../datasets/dataset_3_radiomics.csv'
rad_df = pd.read_csv(rad_path)

# Set patient ID
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]

# Remove features that are constant or string types (if any)
rad_features = rad_df.select_dtypes(include=[np.number])

# Merge clinical target with radiomics
df = usable_clin[['metastasis']].join(rad_features, how='inner')
df = df.dropna(axis=1, how='all') # Drop empty features
df = df.dropna() # Drop empty patients

print(f"Final aligned cohort: {df.shape[0]} patients")
print(f"Class distribution:\\n{df['metastasis'].value_counts()}")

# Remove highly correlated features (optional but good for XGBoost stability)
corr_matrix = df.drop(columns=['metastasis']).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(f"Dropping {len(to_drop)} highly correlated features out of {len(upper.columns)}.")

X = df.drop(columns=['metastasis'] + to_drop)
y = df['metastasis']
"""))

# Cell 3: Setup Evaluation Metrics
cells.append(nbf.v4.new_markdown_cell("""## 2. Evaluation Strategy & Metrics"""))
cells.append(nbf.v4.new_code_cell("""def evaluate_model(y_true, y_prob, threshold=0.5):
    auroc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_curve, precision_curve)
    c_index = concordance_index(y_true, y_prob)
    
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    
    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'C-index': c_index,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'F2': f2
    }
"""))

# Cell 4: Preprocessing & SMOTE
cells.append(nbf.v4.new_markdown_cell("""## 3. Preprocessing & SMOTE Handling"""))
cells.append(nbf.v4.new_code_cell("""# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Apply SMOTE to training set ONLY
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print(f"Training set after SMOTE: {X_train_sm.shape[0]} samples")
"""))

# Cell 5: Modeling
cells.append(nbf.v4.new_markdown_cell("""## 4. Train XGBoost Model"""))
cells.append(nbf.v4.new_code_cell("""model3 = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model3.fit(X_train_sm, y_train_sm)

# Save the model
os.makedirs('../models/dataset_3', exist_ok=True)
model3.save_model('../models/dataset_3/Model3_Imaging_TCGA.json')
joblib.dump(scaler, '../models/dataset_3/Model3_Scaler.pkl')
# Save the feature columns exactly as ordered
joblib.dump(X.columns.tolist(), '../models/dataset_3/Model3_Features.pkl')

print("Model saved to ../models/dataset_3/")

y_prob = model3.predict_proba(X_test_scaled)[:, 1]
"""))

# Cell 6: Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 5. Threshold Optimization & Final Metrics"""))
cells.append(nbf.v4.new_code_cell("""precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find optimal threshold prioritizing Recall >= 0.80
valid_idx = [i for i, r in enumerate(recalls) if r >= 0.80]
if len(valid_idx) > 0:
    optimal_idx = max(valid_idx, key=lambda i: precisions[i])
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
else:
    optimal_threshold = 0.5
    
print(f"Optimal Decision Threshold: {optimal_threshold:.4f}")

results = evaluate_model(y_test, y_prob, threshold=optimal_threshold)
print("\\n--- Model 3 Final Results (Test Set) ---")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Save results
res_df = pd.DataFrame([results], index=['TCGA_Imaging_Model'])
res_df.to_csv('../results/Model3_Performance.csv')
print("\\nResults saved to ../results/Model3_Performance.csv")
"""))

nb.cells = cells
nbf.write(nb, '/home/administrator/Desktop/RCC/notebooks/03_Model3_Imaging_TCGA.ipynb')
print("Notebook 03_Model3_Imaging_TCGA.ipynb generated successfully.")
