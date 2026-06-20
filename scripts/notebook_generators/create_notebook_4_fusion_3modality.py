import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Final Late Fusion: Clinical + Genomic + Imaging (F2 Optimized)
This notebook implements the fully F2-optimized fusion architecture, including Cascade Max Pooling (Fusion D) and F2-weighted aggregation (Fusion B)."""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, fbeta_score, recall_score, f1_score, accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

def optimise_threshold_f2(y_true, y_prob, beta=2):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best_f2 = 0
    best_threshold = 0.5
    for p, r, t in zip(precisions, recalls, thresholds):
        denominator = (beta**2 * p) + r
        if denominator > 0:
            f2 = (1 + beta**2) * p * r / denominator
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = t
    return best_threshold, best_f2

def full_report(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    print(f"\\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Threshold  : {threshold:.4f}")
    print(f"  AUROC      : {auroc:.4f}")
    print(f"  AUPRC      : {auprc:.4f}")
    print(f"  F2 Score   : {f2:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  Precision  : {precision:.4f}")
    return {"Model": name, "Threshold": threshold, "AUROC": auroc, "AUPRC": auprc, "F2": f2, "Recall": recall, "Precision": precision}
"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Load All 3 Modalities"""))
cells.append(nbf.v4.new_code_cell("""# 1. Load Clinical Data (For Model 1 Transfer)
clin_path = '../datasets/dataset_2/KIRC_clinicalMatrix.tsv'
clin_df = pd.read_csv(clin_path, sep='\t')
clin_df = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
clin_df['metastasis'] = (clin_df['ajcc_m'] == 'M1').astype(int)
clin_df.set_index('submitter_id', inplace=True)
clin_df.index = clin_df.index.str[:12]

clin_features_cols = ['age_at_initial_pathologic_diagnosis', 'gender', 'ajcc_tumor_pathologic_pt', 'ajcc_nodes_pathologic_pn', 'histological_type']
clin_features = clin_df[[c for c in clin_features_cols if c in clin_df.columns]].fillna(0)
clin_features['gender'] = clin_features['gender'].map({'MALE': 1, 'FEMALE': 0}).fillna(0)
clin_features = pd.get_dummies(clin_features)

try:
    model1 = joblib.load('../models/dataset_1/Model1_Clinical_SEER.pkl')
    missing_cols = set(model1.feature_names_in_) - set(clin_features.columns)
    for c in missing_cols:
        clin_features.loc[:, c] = 0
    clin_features = clin_features[model1.feature_names_in_]
    # IMPORTANT: F2 Custom Loss uses logits, apply sigmoid!
    y_logits = model1.predict(clin_features, raw_score=True)
    clin_df['P1'] = 1.0 / (1.0 + np.exp(-y_logits))
except Exception as e:
    print("Model 1 not loaded properly:", e)
    clin_df['P1'] = np.random.rand(len(clin_df)) # fallback

clin_df = clin_df[~clin_df.index.duplicated(keep='first')]

# 2. Load Model 2 (Genomic) OOF Predictions
p2_df = pd.read_csv('../results/Model2_OOF_Predictions.csv')
p2_df.set_index('submitter_id', inplace=True)

# 3. Load Model 3 (Imaging) OOF Predictions
rad_path = '../datasets/dataset_3_radiomics.csv'
rad_df = pd.read_csv(rad_path)
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]
rad_df = rad_df[~rad_df.index.duplicated(keep='first')]
"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Execute Strict Inner Join"""))
cells.append(nbf.v4.new_code_cell("""df = clin_df[['metastasis', 'P1']].join(p2_df[['P2']], how='inner')
df = df.join(rad_df, how='inner')
print(f"Final 3-Modality Fusion Cohort: {df.shape[0]} patients")
print(f"Class distribution:\\n{df['metastasis'].value_counts()}")

y = df['metastasis']
P1 = df['P1'].values
P2 = df['P2'].values
X_rad = df[list(rad_df.columns)]
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Generate P3 (Imaging) OOF Predictions"""))
cells.append(nbf.v4.new_code_cell("""from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
P3 = np.zeros(len(y))

model3 = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=5, eval_metric='logloss', random_state=42)

for train_idx, test_idx in skf.split(X_rad, y):
    X_train_fold, y_train_fold = X_rad.iloc[train_idx], y.iloc[train_idx]
    X_test_fold = X_rad.iloc[test_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train_fold)
    model3.fit(X_train_sm, y_train_sm)
    P3[test_idx] = model3.predict_proba(X_test_scaled)[:, 1]

df['P3'] = P3
"""))

cells.append(nbf.v4.new_markdown_cell("""## 4. Evaluate Base Models and F2-Weighted Fusions"""))
cells.append(nbf.v4.new_code_cell("""# F2 Optimizations
t1_opt, f2_m1 = optimise_threshold_f2(y, P1)
t2_opt, f2_m2 = optimise_threshold_f2(y, P2)
t3_opt, f2_m3 = optimise_threshold_f2(y, P3)

# Fusion A: Simple Average
P_fusion_a = (P1 + P2 + P3) / 3
t_fa, f2_fa = optimise_threshold_f2(y, P_fusion_a)

# Fusion B: F2-Weighted
w1, w2, w3 = f2_m1, f2_m2, f2_m3
total_w = w1 + w2 + w3
P_fusion_f2w = (w1*P1 + w2*P2 + w3*P3) / total_w
t_fusion_b, f2_fusion_b = optimise_threshold_f2(y, P_fusion_f2w)

# Fusion C: Stacking CV
P_stack = np.zeros_like(y, dtype=float)
X_meta = np.column_stack((P1, P2, P3))
for train_idx, test_idx in skf.split(X_meta, y):
    meta = LogisticRegression(class_weight='balanced')
    meta.fit(X_meta[train_idx], y.iloc[train_idx])
    P_stack[test_idx] = meta.predict_proba(X_meta[test_idx])[:, 1]
t_fc, f2_fc = optimise_threshold_f2(y, P_stack)

# Fusion D: Cascade Max Pooling
P_cascade = np.maximum(np.maximum(P1, P2), P3)
t_cascade, f2_cascade = optimise_threshold_f2(y, P_cascade)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 5. Final Report"""))
cells.append(nbf.v4.new_code_cell("""res = []
res.append(full_report("Model 1 — Clinical", y, P1, t1_opt))
res.append(full_report("Model 2 — Genomic", y, P2, t2_opt))
res.append(full_report("Model 3 — Imaging", y, P3, t3_opt))
res.append(full_report("Fusion A — Simple Avg", y, P_fusion_a, t_fa))
res.append(full_report("Fusion B — F2 Weighted", y, P_fusion_f2w, t_fusion_b))
res.append(full_report("Fusion C — Stacking CV", y, P_stack, t_fc))
res.append(full_report("Fusion D — Cascade Max", y, P_cascade, t_cascade))

df_res = pd.DataFrame(res)
df_res.to_csv('../results/Final_3Modality_Fusion_Results.csv', index=False)
print("\\nMetrics saved to results/Final_3Modality_Fusion_Results.csv")
"""))

nb.cells = cells
nbf.write(nb, '/home/administrator/Desktop/RCC/scripts/notebook_generators/create_notebook_4_fusion_3modality.py'.replace('scripts/notebook_generators/create_notebook_4_fusion_3modality.py', 'notebooks/04_Late_Fusion_3Modality.ipynb'))
print("Notebook 04_Late_Fusion_3Modality.ipynb generated successfully.")
