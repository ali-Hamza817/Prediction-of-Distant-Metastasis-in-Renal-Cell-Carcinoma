import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc

def f2_weighted_loss(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
    fn_weight = 4.0
    fp_weight = 1.0
    grad = -(fn_weight * y_true * (1 - y_pred) - fp_weight * (1 - y_true) * y_pred)
    hess = (fn_weight * y_true * y_pred * (1 - y_pred) + fp_weight * (1 - y_true) * y_pred * (1 - y_pred))
    return grad, hess

model1 = joblib.load('../models/dataset_1/Model1_Clinical_SEER.pkl')
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
clin_df['P1'] = 1.0 / (1.0 + np.exp(-model1.predict(clin_features.values, raw_score=True)))

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

fpr, tpr, _ = roc_curve(y_tcga, P_fusion_a)
print("Fusion A AUC calculated by Python:", auc(fpr, tpr))
fpr, tpr, _ = roc_curve(y_tcga, P1_tcga)
print("Model 1 TCGA AUC calculated by Python:", auc(fpr, tpr))
fpr, tpr, _ = roc_curve(y_tcga, P2)
print("Model 2 TCGA AUC calculated by Python:", auc(fpr, tpr))
fpr, tpr, _ = roc_curve(y_tcga, P3)
print("Model 3 TCGA AUC calculated by Python:", auc(fpr, tpr))
