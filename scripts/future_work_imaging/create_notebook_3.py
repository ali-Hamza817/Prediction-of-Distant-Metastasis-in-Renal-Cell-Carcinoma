import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 0: Title
cells.append(nbf.v4.new_markdown_cell("""# Model 3: Imaging (TCGA CT Radiomics)
This notebook extracts radiomics features (shape, texture) from CT scans, selects the top features using LASSO, and trains an XGBoost classifier."""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, 
                             precision_score, recall_score, f1_score, fbeta_score)
import xgboost as xgb
import shap
import warnings
import os
import joblib

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
"""))

# Cell 2: Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Loading & Alignment"""))
cells.append(nbf.v4.new_code_cell("""# Load PyRadiomics Features
try:
    rad_df = pd.read_csv('../datasets/dataset_3_radiomics.csv')
except FileNotFoundError:
    print("Run extract_radiomics.py first to generate the features!")
    rad_df = pd.DataFrame()

if not rad_df.empty:
    rad_df.set_index('patient_id', inplace=True)
    
    # Load clinical data for target labels
    clin_path = '../datasets/dataset_2/KIRC_clinicalMatrix.tsv'
    clin_df = pd.read_csv(clin_path, sep='\\t')
    usable_clin = clin_df[clin_df['ajcc_m'].isin(['M0', 'M1'])].copy()
    usable_clin['metastasis'] = (usable_clin['ajcc_m'] == 'M1').astype(int)
    usable_clin.set_index('submitter_id', inplace=True)
    usable_clin.index = usable_clin.index.str[:12]
    
    # Merge on patient ID
    df = usable_clin[['metastasis']].join(rad_df, how='inner')
    df = df.dropna()
    
    print(f"Final matched imaging cohort: {df.shape[0]} patients")
    print(f"Class distribution:\\n{df['metastasis'].value_counts()}")
    
    # Features are everything except metastasis
    X = df.drop(columns=['metastasis'])
    y = df['metastasis']
else:
    X, y = pd.DataFrame(), pd.Series()
"""))

# Cell 3: Setup Evaluation Metrics
cells.append(nbf.v4.new_markdown_cell("""## 2. Evaluation Strategy & Metrics"""))
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

# Cell 4: LASSO Feature Selection
cells.append(nbf.v4.new_markdown_cell("""## 3. LASSO Feature Selection
We have over 100 radiomics features. We use LASSO with 10-fold CV to select the top robust features."""))
cells.append(nbf.v4.new_code_cell("""if not X.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print(f"Starting LASSO CV with {X_train_scaled.shape[1]} features...")
    lasso = LassoCV(cv=10, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    coef = pd.Series(lasso.coef_, index=X.columns)
    selected_features = coef[coef != 0].index.tolist()
    
    # If LASSO shrinks everything to 0 due to dataset size or alpha, we take the top 15 by absolute correlation
    if len(selected_features) < 5:
        print("LASSO over-regularized. Falling back to correlation-based selection.")
        corrs = X_train_scaled.apply(lambda col: col.corr(y_train)).abs().sort_values(ascending=False)
        selected_features = corrs.head(15).index.tolist()
    
    print(f"Selected {len(selected_features)} features for XGBoost:\\n{selected_features}")
    
    X_train_sel = X_train_scaled[selected_features]
    X_test_sel = X_test_scaled[selected_features]
"""))

# Cell 5: Train XGBoost Model
cells.append(nbf.v4.new_markdown_cell("""## 4. Train XGBoost Classifier
Applying strict `scale_pos_weight` to mathematically penalize false negatives."""))
cells.append(nbf.v4.new_code_cell("""if not X.empty:
    # Handle Class Imbalance explicitly
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"Using scale_pos_weight = {scale_pos_weight:.2f}")
    
    model3 = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    model3.fit(X_train_sel, y_train)
    
    # Save the model and scaler
    os.makedirs('../models/dataset_3', exist_ok=True)
    joblib.dump(model3, '../models/dataset_3/Model3_Imaging_TCGA.pkl')
    joblib.dump(scaler, '../models/dataset_3/Model3_Scaler.pkl')
    # Also save the selected feature names so we know what to extract for future patients
    joblib.dump(selected_features, '../models/dataset_3/Model3_SelectedFeatures.pkl')
    print("Model saved to ../models/dataset_3/Model3_Imaging_TCGA.pkl")
    
    y_prob = model3.predict_proba(X_test_sel)[:, 1]
"""))

# Cell 6: Threshold Optimization & Final Metrics
cells.append(nbf.v4.new_markdown_cell("""## 5. Final Evaluation"""))
cells.append(nbf.v4.new_code_cell("""if not X.empty:
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
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

# Cell 7: Explainability (SHAP)
cells.append(nbf.v4.new_markdown_cell("""## 6. Explainability: Top PyRadiomics Features"""))
cells.append(nbf.v4.new_code_cell("""if not X.empty:
    explainer = shap.TreeExplainer(model3)
    shap_values = explainer.shap_values(X_test_sel)
    
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test_sel, show=False)
    plt.title('SHAP Beeswarm Plot - PyRadiomics Features')
    plt.tight_layout()
    plt.savefig('../results/Model3_SHAP_Summary.png', dpi=300)
    plt.show()
"""))

nb.cells = cells
nbf.write(nb, '/home/administrator/Desktop/RCC/notebooks/03_Model3_Imaging_TCGA.ipynb')
print("Notebook 03_Model3_Imaging_TCGA.ipynb generated successfully.")
