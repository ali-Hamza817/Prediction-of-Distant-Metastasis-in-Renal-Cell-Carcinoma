import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, fbeta_score, recall_score, precision_recall_curve

# Load predictions from notebooks (we can re-run the fusion logic)
# Actually, since I have the models and data, I can just recalculate it quickly.
# I will just extract the arrays from the generated fusion code if possible, or rewrite it.
import joblib

df = pd.read_csv('datasets/dataset_3_radiomics.csv')
clin_df = pd.read_csv('datasets/dataset_2/KIRC_clinicalMatrix.tsv', sep='\t')
clin_df = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
clin_df.set_index('submitter_id', inplace=True)
clin_df.index = clin_df.index.str[:12]

gen_path = 'datasets/dataset_2/HiSeqV2.gz'
gen_df = pd.read_csv(gen_path, sep='\t', index_col=0).T
gen_df.index = gen_df.index.str[:12]

rad_df = pd.read_csv('datasets/dataset_3_radiomics.csv')
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]

clin_df = clin_df[~clin_df.index.duplicated(keep='first')]
gen_df = gen_df[~gen_df.index.duplicated(keep='first')]
rad_df = rad_df[~rad_df.index.duplicated(keep='first')]

# Selected Genes from Model 2
genes = ['FKBP15', 'SLC31A1', 'CPT2', 'PATJ', 'CALR']
available_genes = [g for g in genes if g in gen_df.columns]
gen_features = gen_df[available_genes]

# Clinical features Model 1
clin_features_cols = ['age_at_initial_pathologic_diagnosis', 'gender', 'ajcc_tumor_pathologic_pt', 'ajcc_nodes_pathologic_pn', 'histological_type']
clin_features = clin_df[[c for c in clin_features_cols if c in clin_df.columns]]
clin_features['gender'] = clin_features['gender'].map({'MALE': 1, 'FEMALE': 0})
# Fill missing
clin_features = clin_features.fillna(0)
# Dummies
clin_features = pd.get_dummies(clin_features)
# Ensure columns match Model 1
try:
    model1 = joblib.load('models/dataset_1/Model1_Clinical_SEER.pkl')
    # Pad columns to match model1
    missing_cols = set(model1.feature_names_in_) - set(clin_features.columns)
    for c in missing_cols:
        clin_features.loc[:, c] = 0
    clin_features = clin_features[model1.feature_names_in_]
except Exception as e:
    pass

df_final = pd.concat([clin_df[['ajcc_m']], clin_features, gen_features, rad_df], axis=1, join='inner')
y = df_final['ajcc_m'].map({'M1': 1, 'M0': 0}).values
clin_X = df_final[model1.feature_names_in_]
gen_X = df_final[available_genes]
rad_X = df_final[rad_df.columns]

# Predictions
from sklearn.preprocessing import StandardScaler
model2 = joblib.load('models/dataset_2/Model2_Genomic_TCGA.pkl')
model2_scaler = joblib.load('models/dataset_2/Model2_Scaler.pkl')
gen_X_scaled = model2_scaler.transform(gen_X)

from xgboost import XGBClassifier
model3 = XGBClassifier()
model3.load_model('models/dataset_3/Model3_Imaging_TCGA.json')
model3_scaler = joblib.load('models/dataset_3/Model3_Scaler.pkl')
rad_features = joblib.load('models/dataset_3/Model3_Features.pkl')
rad_X = df_final[rad_features]
rad_X_scaled = model3_scaler.transform(rad_X)

P1 = model1.predict_proba(clin_X)[:, 1]
P2 = model2.predict_proba(gen_X_scaled)[:, 1]
P3 = model3.predict_proba(rad_X_scaled)[:, 1]

# Simple Average
P_avg = (P1 + P2 + P3) / 3

# Stacking Meta Learner CV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_meta = np.column_stack((P1, P2, P3))
P_stack = np.zeros_like(y, dtype=float)
for train_idx, test_idx in cv.split(X_meta, y):
    meta = LogisticRegression(class_weight='balanced')
    meta.fit(X_meta[train_idx], y[train_idx])
    P_stack[test_idx] = meta.predict_proba(X_meta[test_idx])[:, 1]

# Function to get metrics optimally
def get_metrics(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-8)
    best_idx = np.argmax(f2_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_prob >= best_thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    rec = recall_score(y_true, y_pred)
    return acc, auc, ap, f1, f2, rec

results = []
results.append(("Model 1: Clinical (SEER)", *get_metrics(y, P1)))
results.append(("Model 2: Genomic (TCGA)", *get_metrics(y, P2)))
results.append(("Model 3: Imaging (TCGA)", *get_metrics(y, P3)))
results.append(("Fusion A: Simple avg", *get_metrics(y, P_avg)))
results.append(("Fusion C: Stacking", *get_metrics(y, P_stack)))

df_res = pd.DataFrame(results, columns=["Model", "Accuracy", "AUROC", "AUPRC", "F1", "F2", "Recall"])
print(df_res.to_string(index=False))
df_res.to_csv("results/Final_3Modality_Extended.csv", index=False)
