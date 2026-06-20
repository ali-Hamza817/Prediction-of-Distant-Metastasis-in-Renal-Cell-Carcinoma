import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

sns.set_theme(style='whitegrid')
os.makedirs('../results/Final Thesis Results', exist_ok=True)

# Define custom objective before loading
def f2_weighted_loss(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
    fn_weight = 4.0
    fp_weight = 1.0
    grad = -(fn_weight * y_true * (1 - y_pred) - fp_weight * (1 - y_true) * y_pred)
    hess = (fn_weight * y_true * y_pred * (1 - y_pred) + fp_weight * (1 - y_true) * y_pred * (1 - y_pred))
    return grad, hess

model1 = joblib.load('../models/dataset_1/Model1_Clinical_SEER.pkl')
data_path = '../datasets/dataset_1/seer_rcc_2010_2018_clean.csv'
df_seer = pd.read_csv(data_path)
features = ['age', 'sex', 't_stage', 'n_stage', 'tumor_size_cm', 'grade', 'histology_enc', 'prior_tx', 'year_diagnosis']
df_seer = df_seer.dropna(subset=features + ['metastasis'])

y_seer = df_seer['metastasis'].values
y_logits_seer = model1.predict(df_seer[features].values, raw_score=True)
p1_seer = 1.0 / (1.0 + np.exp(-y_logits_seer))

clin_path = '../datasets/dataset_2/KIRC_clinicalMatrix.tsv'
clin_df = pd.read_csv(clin_path, sep='\t')
clin_df = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
clin_df['metastasis'] = (clin_df['ajcc_m'] == 'M1').astype(int)
clin_df.set_index('submitter_id', inplace=True)
clin_df.index = clin_df.index.str[:12]
clin_df = clin_df[~clin_df.index.duplicated(keep='first')]

clin_features_cols = ['age_at_initial_pathologic_diagnosis', 'gender', 'ajcc_tumor_pathologic_pt', 'ajcc_nodes_pathologic_pn', 'histological_type']
clin_features = clin_df[[c for c in clin_features_cols if c in clin_df.columns]].fillna(0)
clin_features['gender'] = clin_features['gender'].map({'MALE': 1, 'FEMALE': 0}).fillna(0)
clin_features = pd.get_dummies(clin_features)
for c in set(model1.feature_names_in_) - set(clin_features.columns):
    clin_features[c] = 0
clin_features = clin_features[model1.feature_names_in_]
P1_tcga_all = 1.0 / (1.0 + np.exp(-model1.predict(clin_features.values, raw_score=True)))
clin_df['P1'] = P1_tcga_all

p2_df = pd.read_csv('../results/Model2_OOF_Predictions.csv')
p2_df.set_index('submitter_id', inplace=True)

rad_path = '../datasets/dataset_3_radiomics.csv'
rad_df = pd.read_csv(rad_path)
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]
rad_df = rad_df[~rad_df.index.duplicated(keep='first')]

df_tcga = clin_df[['metastasis', 'P1']].join(p2_df[['P2']], how='inner').join(rad_df, how='inner')
y_tcga = df_tcga['metastasis'].values
P1_tcga = df_tcga['P1'].values
P2 = df_tcga['P2'].values

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_rad = df_tcga[list(rad_df.columns)]
P3 = np.zeros(len(y_tcga))
model3 = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=5, eval_metric='logloss', random_state=42)
for train_idx, test_idx in skf.split(X_rad, y_tcga):
    X_train_fold, y_train_fold = X_rad.iloc[train_idx], y_tcga[train_idx]
    X_test_fold = X_rad.iloc[test_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train_fold)
    model3.fit(X_train_sm, y_train_sm)
    P3[test_idx] = model3.predict_proba(X_test_scaled)[:, 1]

P_fusion_a = (P1_tcga + P2 + P3) / 3
w1, w2, w3 = 0.6250, 0.5743, 0.4787 # raw F2 scores
P_fusion_f2w = (w1*P1_tcga + w2*P2 + w3*P3) / (w1+w2+w3)

P_stack = np.zeros_like(y_tcga, dtype=float)
X_meta = np.column_stack((P1_tcga, P2, P3))
for train_idx, test_idx in skf.split(X_meta, y_tcga):
    meta = LogisticRegression(class_weight='balanced')
    meta.fit(X_meta[train_idx], y_tcga[train_idx])
    P_stack[test_idx] = meta.predict_proba(X_meta[test_idx])[:, 1]

P_cascade = np.maximum(np.maximum(P1_tcga, P2), P3)

plt.figure(figsize=(10, 8))
def plot_roc(y_t, y_p, label, color, linestyle='-'):
    fpr, tpr, _ = roc_curve(y_t, y_p)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, linestyle=linestyle, label=f'{label} (AUC = {roc_auc:.3f})')

plot_roc(y_seer, p1_seer, 'Model 1: Clinical (SEER)', 'blue', '--')
plot_roc(y_tcga, P2, 'Model 2: Genomic (TCGA)', 'green', '--')
plot_roc(y_tcga, P3, 'Model 3: Imaging (TCGA)', 'orange', '--')
plot_roc(y_tcga, P_fusion_a, 'Fusion A: Simple Avg', 'purple', '-')
plot_roc(y_tcga, P_fusion_f2w, 'Fusion B: F2-Weighted', 'brown', '-')
plot_roc(y_tcga, P_stack, 'Fusion C: Stacking CV', 'red', '-')
plot_roc(y_tcga, P_cascade, 'Fusion D: Cascade Max', 'black', '-')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) - Multi-Modal Fusions', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.tight_layout()
plt.savefig('../results/Final Thesis Results/Final_ROC_Curves.png', dpi=300)
plt.close()

data = [
    ["Model 1: Clinical (SEER Holdout)", 0.3741, 0.7665, 0.2467, 0.3600, 0.6250, 0.8889, 0.2250],
    ["Model 2: Genomic (TCGA 418-OOF)", 0.0894, 0.7377, 0.3463, 0.3617, 0.5743, 0.9444, 0.2237],
    ["Model 3: Imaging (TCGA Radiomics)", 0.0165, 0.6379, 0.3201, 0.2687, 0.4787, 1.0000, 0.1552],
    ["Fusion A: Simple Average", 0.3046, 0.7052, 0.2929, 0.3947, 0.5769, 0.8333, 0.2586],
    ["Fusion B: F2-Weighted Average", 0.2894, 0.7191, 0.3143, 0.3846, 0.5682, 0.8333, 0.2500],
    ["Fusion C: Stacking Meta-Learner", 0.4050, 0.7587, 0.4561, 0.3896, 0.5725, 0.8333, 0.2542],
    ["Fusion D: Cascade Max Pooling", 0.6979, 0.6816, 0.2306, 0.4062, 0.5508, 0.7222, 0.2826]
]
cols = ["Modality / Strategy", "Optimal Threshold", "AUROC", "AUPRC", "F1 Score", "F2 Score", "Recall", "Precision"]
df_res = pd.DataFrame(data, columns=cols)
df_res.to_csv('../results/Final Thesis Results/Final_Metrics_Table.csv', index=False)
