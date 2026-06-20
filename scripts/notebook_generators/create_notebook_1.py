import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 0: Title
cells.append(nbf.v4.new_markdown_cell("""# Model 1: Clinical (SEER Dataset) - F2 Optimized
This notebook trains a LightGBM model on the SEER dataset (Dataset 1) using a custom F2-weighted loss function and Optuna hyperparameter search directly maximizing the F2 score."""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, 
                             precision_score, recall_score, f1_score, fbeta_score)
from imblearn.over_sampling import SMOTE
import warnings
import os
import joblib

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
optuna.logging.set_verbosity(optuna.logging.WARNING)
"""))

# Cell 2: Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Loading & Alignment"""))
cells.append(nbf.v4.new_code_cell("""# Load the SEER dataset
data_path = '../datasets/dataset_1/seer_rcc_2010_2018_clean.csv'
df = pd.read_csv(data_path)

print(f"Loaded Dataset 1: {df.shape[0]} patients, {df.shape[1]} columns")

# Define features and targets
features = ['age', 'sex', 't_stage', 'n_stage', 'tumor_size_cm', 'grade', 'histology_enc', 'prior_tx', 'year_diagnosis']
main_target = 'metastasis'

# Drop patients with missing values in core features (if any)
df = df.dropna(subset=features + [main_target])

X = df[features].values
y = df[main_target].values

print(f"Overall Metastasis Class Distribution:\\n{df[main_target].value_counts(normalize=True)*100}")
"""))

# Cell 3: F2 Loss and Threshold Opt
cells.append(nbf.v4.new_markdown_cell("""## 2. Custom Loss Function & Evaluation Metrics"""))
cells.append(nbf.v4.new_code_cell("""def f2_weighted_loss(y_true, y_pred):
    \"\"\"Custom loss function that penalises false negatives 4x more than false positives.\"\"\"
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
    fn_weight = 4.0
    fp_weight = 1.0
    grad = -(fn_weight * y_true * (1 - y_pred) - fp_weight * (1 - y_true) * y_pred)
    hess = (fn_weight * y_true * y_pred * (1 - y_pred) + fp_weight * (1 - y_true) * y_pred * (1 - y_pred))
    return grad, hess

def f2_eval_metric(y_true, y_pred):
    # LightGBM custom loss passes raw logits to eval metric, so we apply sigmoid
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    f2 = fbeta_score(y_true, y_pred_binary, beta=2, zero_division=0)
    return 'f2_score', f2, True

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
"""))

# Cell 4: Optuna F2 Search
cells.append(nbf.v4.new_markdown_cell("""## 3. Optuna Hyperparameter Search directly optimizing F2"""))
cells.append(nbf.v4.new_code_cell("""def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'learning_rate':     trial.suggest_float('lr', 0.005, 0.1, log=True),
        'max_depth':         trial.suggest_int('max_depth', 3, 7),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 30),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample', 0.5, 1.0),
        'reg_alpha':         trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('lambda', 1e-4, 10.0, log=True),
        'scale_pos_weight':  trial.suggest_float('spw', 5.0, 20.0),
        'random_state': 42, 
        'verbosity': -1,
    }
    smote_ratio = trial.suggest_float('smote_ratio', 0.3, 1.0)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_f2s = []
    
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        # SMOTE inside fold
        sm = SMOTE(sampling_strategy=smote_ratio, random_state=42, k_neighbors=3)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        
        model = lgb.LGBMClassifier(**params)
        model.set_params(objective=f2_weighted_loss)
        
        # Fit with custom eval metric
        model.fit(
            X_tr_sm, y_tr_sm,
            eval_set=[(X_val, y_val)],
            eval_metric=f2_eval_metric,
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # We need to manually compute raw probabilities since custom loss outputs logits
        y_logits = model.predict(X_val, raw_score=True)
        y_prob = 1.0 / (1.0 + np.exp(-y_logits))
        
        best_t, best_f2 = optimise_threshold_f2(y_val, y_prob)
        fold_f2s.append(best_f2)
    
    return np.mean(fold_f2s)

print("Starting Optuna Study (50 trials for time constraints)...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\\nBest F2 from Optuna : {study.best_value:.4f}")
print(f"Best parameters     : {study.best_params}")
"""))

# Cell 5: Train Final Model
cells.append(nbf.v4.new_markdown_cell("""## 4. Train Final Model & Extract OOF Predictions"""))
cells.append(nbf.v4.new_code_cell("""# Get best params
best_params = study.best_params
smote_ratio = best_params.pop('smote_ratio')
best_params['random_state'] = 42
best_params['verbosity'] = -1
# Need to rename keys back for LightGBM
if 'lr' in best_params: best_params['learning_rate'] = best_params.pop('lr')
if 'alpha' in best_params: best_params['reg_alpha'] = best_params.pop('alpha')
if 'lambda' in best_params: best_params['reg_lambda'] = best_params.pop('lambda')
if 'spw' in best_params: best_params['scale_pos_weight'] = best_params.pop('spw')

# Train final model on full dataset
sm = SMOTE(sampling_strategy=smote_ratio, random_state=42, k_neighbors=3)
X_sm, y_sm = sm.fit_resample(X, y)
X_sm_df = pd.DataFrame(X_sm, columns=features)

final_model = lgb.LGBMClassifier(**best_params)
final_model.set_params(objective=f2_weighted_loss)
final_model.fit(X_sm_df, y_sm)

os.makedirs('../models/dataset_1', exist_ok=True)
joblib.dump(final_model, '../models/dataset_1/Model1_Clinical_SEER.pkl')
print("Model saved to ../models/dataset_1/Model1_Clinical_SEER.pkl")

# Generate probabilities for evaluation on full set
y_logits = final_model.predict(X, raw_score=True)
y_prob = 1.0 / (1.0 + np.exp(-y_logits))

best_thresh, final_f2 = optimise_threshold_f2(y, y_prob)
print(f"Final Model Threshold: {best_thresh:.4f}")
print(f"Final Model F2: {final_f2:.4f}")

df['P1'] = y_prob
# We don't save OOF for Model 1 because it's transferred directly. 
# But we save its performance.
"""))

nb.cells = cells
nbf.write(nb, '/home/administrator/Desktop/RCC/scripts/notebook_generators/create_notebook_1.py'.replace('scripts/notebook_generators/create_notebook_1.py', 'notebooks/01_Model1_Clinical_SEER.ipynb'))
print("Notebook 01_Model1_Clinical_SEER.ipynb generated successfully.")
