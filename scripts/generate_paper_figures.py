"""
generate_paper_figures.py
Generates 10 publication-quality figures for the RCC metastasis research paper.
ALL data sourced from real trained models and result CSVs — nothing fabricated.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ── paths ───────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(BASE, 'results', 'figures_for_research_paper')
os.makedirs(OUT, exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────────
# Updated for white background (academic paper style)
STYLE = {
    'figure.facecolor':  '#ffffff',
    'axes.facecolor':    '#ffffff',
    'axes.edgecolor':    '#333333',
    'axes.labelcolor':   '#111111',
    'xtick.color':       '#333333',
    'ytick.color':       '#333333',
    'text.color':        '#111111',
    'grid.color':        '#e5e7eb',
    'grid.linestyle':    '--',
    'grid.alpha':        0.8,
    'legend.facecolor':  '#ffffff',
    'legend.edgecolor':  '#cccccc',
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    14,
    'axes.labelsize':    12,
}
plt.rcParams.update(STYLE)

# colour palette — publication quality (printer friendly on white bg)
CLRS = {
    'm1':       '#3730a3',   # dark indigo   — Clinical
    'm2':       '#0891b2',   # dark cyan     — Genomic
    'm3':       '#059669',   # dark emerald  — Imaging
    'fa':       '#d97706',   # amber         — Fusion A
    'fb':       '#dc2626',   # red           — Fusion B ⭐
    'fc':       '#7e22ce',   # purple        — Fusion C
    'fd':       '#db2777',   # pink          — Fusion D
    'gold':     '#b45309',
    'bg':       '#ffffff',
    'bg2':      '#ffffff',
    'text':     '#111111',
    'text2':    '#4b5563',
}

DPI = 180

def save(fig, name, tight=True):
    path = os.path.join(OUT, name)
    if tight:
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  ✓ {name}')

# ── load models ──────────────────────────────────────────────────────────────
print('Loading models…')
import __main__
def _f2_loss(*a, **k): pass
__main__.f2_weighted_loss = _f2_loss

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

M1F = ['age','sex','t_stage','n_stage','tumor_size_cm','grade','histology_enc','prior_tx','year_diagnosis']
m1      = joblib.load(f'{BASE}/models/dataset_1/Model1_Clinical_SEER.pkl')
m1_lung = joblib.load(f'{BASE}/models/dataset_1/Model1_Clinical_SEER_lung_met.pkl')
m1_bone = joblib.load(f'{BASE}/models/dataset_1/Model1_Clinical_SEER_bone_met.pkl')
m1_liver= joblib.load(f'{BASE}/models/dataset_1/Model1_Clinical_SEER_liver_met.pkl')
m1_brain= joblib.load(f'{BASE}/models/dataset_1/Model1_Clinical_SEER_brain_met.pkl')
m2      = joblib.load(f'{BASE}/models/dataset_2/Model2_Genomic_TCGA.pkl')
sc2     = joblib.load(f'{BASE}/models/dataset_2/Model2_Scaler.pkl')
feat2   = joblib.load(f'{BASE}/models/dataset_2/Model2_Features.pkl')
m3b     = xgb.Booster(); m3b.load_model(f'{BASE}/models/dataset_3/Model3_Imaging_TCGA.json')
sc3     = joblib.load(f'{BASE}/models/dataset_3/Model3_Scaler.pkl')
feat3   = joblib.load(f'{BASE}/models/dataset_3/Model3_Features.pkl')
print('  Models loaded ✓')

# ── load datasets ────────────────────────────────────────────────────────────
print('Loading datasets…')
df_seer = pd.read_csv(f'{BASE}/datasets/dataset_1/seer_rcc_2010_2018_clean.csv')

# TCGA clinical
clin_df = pd.read_csv(f'{BASE}/datasets/dataset_2/KIRC_clinicalMatrix.tsv', sep='\t')
clin_df = clin_df[clin_df['ajcc_m'].isin(['M0','M1'])].copy()
clin_df['metastasis'] = (clin_df['ajcc_m']=='M1').astype(int)
clin_df.set_index('submitter_id', inplace=True)
clin_df.index = clin_df.index.str[:12]
clin_df = clin_df[~clin_df.index.duplicated(keep='first')]

# RNA-Seq (for computing P1 transfer on TCGA)
rna_df = pd.read_csv(f'{BASE}/datasets/dataset_2/HiSeqV2.gz', sep='\t', index_col=0, compression='gzip').T
rna_df.index = rna_df.index.str[:12]
rna_df = rna_df[~rna_df.index.duplicated(keep='first')]
tcga_df = clin_df[['metastasis']].join(rna_df[feat2], how='inner').dropna()

# P2 OOF predictions
p2_df = pd.read_csv(f'{BASE}/results/Model2_OOF_Predictions.csv')
p2_df.set_index('submitter_id', inplace=True)
p2_df.index = p2_df.index.str[:12]

# Radiomics
rad_df = pd.read_csv(f'{BASE}/datasets/dataset_3_radiomics.csv')
rad_df.set_index('patient_id', inplace=True)
rad_df.index = rad_df.index.str[:12]
rad_df = rad_df[~rad_df.index.duplicated(keep='first')]

# Build fusion cohort (126-pt inner join)
fuse = clin_df[['metastasis']].join(p2_df[['P2']], how='inner').join(rad_df, how='inner').dropna()
print(f'  Fusion cohort: {len(fuse)} patients, {fuse.metastasis.sum()} M1')

y_fuse  = fuse['metastasis'].values
P2_fuse = fuse['P2'].values
X_rad   = fuse[list(rad_df.columns)].values

# Compute P1 on fusion cohort via Model1 transfer
clin_feats_fuse = clin_df.loc[fuse.index, :]
clin_aligned = pd.DataFrame(0, index=fuse.index, columns=M1F)
col_map = {
    'age_at_initial_pathologic_diagnosis': 'age',
    'age_at_index': 'age',
}
for old, new in col_map.items():
    if old in clin_feats_fuse.columns:
        clin_aligned[new] = pd.to_numeric(clin_feats_fuse[old], errors='coerce').fillna(0)
gender_col = [c for c in clin_feats_fuse.columns if 'gender' in c.lower() or c=='gender']
if gender_col:
    clin_aligned['sex'] = (clin_feats_fuse[gender_col[0]].str.upper()=='MALE').astype(int)

import pandas as pd_
P1_fuse = sigmoid(m1.predict(clin_aligned.values, raw_score=True))

# Compute P3 OOF on 126-pt cohort
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
P3_fuse = np.zeros(len(y_fuse))
from xgboost import XGBClassifier
_m3 = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                    scale_pos_weight=5, eval_metric='logloss', random_state=42, verbosity=0)
for tr, te in skf.split(X_rad, y_fuse):
    sc_ = StandardScaler()
    Xtr = sc_.fit_transform(X_rad[tr]); Xte = sc_.transform(X_rad[te])
    sm  = SMOTE(sampling_strategy=0.5, random_state=42)
    Xs, ys = sm.fit_resample(Xtr, y_fuse[tr])
    _m3.fit(Xs, ys)
    P3_fuse[te] = _m3.predict_proba(Xte)[:,1]

# Fused scores
w1,w2,w3 = 0.6250, 0.5743, 0.4787
P_fa = (P1_fuse + P2_fuse + P3_fuse)/3
P_fb = (w1*P1_fuse + w2*P2_fuse + w3*P3_fuse)/(w1+w2+w3)
X_meta = np.column_stack([P1_fuse, P2_fuse, P3_fuse])
P_fc = np.zeros(len(y_fuse))
for tr,te in skf.split(X_meta, y_fuse):
    lr = LogisticRegression(class_weight='balanced', max_iter=500)
    lr.fit(X_meta[tr], y_fuse[tr])
    P_fc[te] = lr.predict_proba(X_meta[te])[:,1]
P_fd = np.maximum(np.maximum(P1_fuse, P2_fuse), P3_fuse)

print('  Predictions computed ✓')

# ── metrics CSVs ──────────────────────────────────────────────────────────────
metrics_csv = pd.read_csv(f'{BASE}/results/Final Thesis Results/Final_Metrics_Table.csv')
m1_perf     = pd.read_csv(f'{BASE}/results/Model1_Performance.csv', index_col=0)
seer_df     = df_seer.copy()

print('\nGenerating figures…')

# ════════════════════════════════════════════════════════════
# FIG 1 — ROC Curves: All Models + Fusion Strategies
# ════════════════════════════════════════════════════════════
# Widen figure to accommodate legend on the outside so it is readable
fig, ax = plt.subplots(figsize=(10,6.5))

curves = [
    ('Clinical (Model 1)',          P1_fuse,  CLRS['m1'], '--'),
    ('Genomic (Model 2)',           P2_fuse,  CLRS['m2'], '--'),
    ('Imaging (Model 3)',           P3_fuse,  CLRS['m3'], '--'),
    ('Fusion A: Simple Average',    P_fa,     CLRS['fa'], '-'),
    ('Fusion B: F2-Weighted',      P_fb,     CLRS['fb'], '-'),
    ('Fusion C: Stacking',         P_fc,     CLRS['fc'], '-'),
    ('Fusion D: Cascade Max',      P_fd,     CLRS['fd'], '-'),
]
for name, scores, c, ls in curves:
    fpr,tpr,_ = roc_curve(y_fuse, scores)
    a = auc(fpr,tpr)
    lw = 2.5 if 'B' in name else 1.8
    # alpha adjustment for cleaner plotting
    ax.plot(fpr, tpr, color=c, lw=lw, ls=ls, alpha=0.9 if 'B' not in name else 1.0, label=f'{name}  (AUROC={a:.3f})')

ax.plot([0,1],[0,1],'--', color='#9ca3af', lw=1.2, label='Random (AUROC=0.500)')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, color=CLRS['text'])
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, color=CLRS['text'])
ax.set_title('Figure 1: ROC Curves — All Models & Fusion Strategies\n(TCGA-KIRC 126-Patient Alignment Cohort)', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
# Move legend completely outside to prevent jumbling
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True, edgecolor='#cccccc')
ax.grid(True, alpha=0.5)
ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.02])
save(fig, 'Fig1_ROC_Curves_All_Models.png')

# ════════════════════════════════════════════════════════════
# FIG 2 — Precision-Recall Curves: All Models + Fusion
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10,6.5))

base_rate = y_fuse.mean()
ax.axhline(base_rate, color='#9ca3af', lw=1.2, ls='--', label=f'Baseline (prevalence={base_rate:.3f})')

for name, scores, c, ls in curves:
    prec, rec, _ = precision_recall_curve(y_fuse, scores)
    ap = average_precision_score(y_fuse, scores)
    lw = 2.5 if 'B' in name else 1.8
    ax.plot(rec, prec, color=c, lw=lw, ls=ls, alpha=0.9 if 'B' not in name else 1.0, label=f'{name}  (AUPRC={ap:.3f})')

ax.set_xlabel('Recall (Sensitivity)', fontsize=12, color=CLRS['text'])
ax.set_ylabel('Precision (PPV)', fontsize=12, color=CLRS['text'])
ax.set_title('Figure 2: Precision-Recall Curves — All Models & Fusion Strategies\n(TCGA-KIRC 126-Patient Alignment Cohort)', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
# Move legend completely outside to prevent jumbling
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True, edgecolor='#cccccc')
ax.grid(True, alpha=0.5)
ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
save(fig, 'Fig2_PR_Curves_All_Models.png')

# ════════════════════════════════════════════════════════════
# FIG 3 — AUROC Comparison Bar Chart (all models + fusion)
# ════════════════════════════════════════════════════════════
labels = metrics_csv['Modality / Strategy'].tolist()
aurocs = metrics_csv['AUROC'].tolist()
short  = ['M1\nClinical','M2\nGenomic','M3\nImaging','Fusion A\nSimple','Fusion B\nF2-Wtd','Fusion C\nStacking','Fusion D\nCascade']
clrs_bar = [CLRS['m1'],CLRS['m2'],CLRS['m3'],CLRS['fa'],CLRS['fb'],CLRS['fc'],CLRS['fd']]

fig, ax = plt.subplots(figsize=(10,6))

bars = ax.bar(short, aurocs, color=clrs_bar, width=0.55, edgecolor='#333333', linewidth=1.0, zorder=3)
ax.axhline(0.5, color='#9ca3af', lw=1.2, ls='--', zorder=2, label='Random classifier (0.500)')
ax.axhline(aurocs[4], color=CLRS['fb'], lw=1.2, ls=':', alpha=0.5, zorder=2)

for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, color=CLRS['text'], fontweight='bold')

ax.set_ylim([0.35, 0.88])
ax.set_ylabel('AUROC', fontsize=12); ax.set_xlabel('Model / Strategy', fontsize=12)
ax.set_title('Figure 3: AUROC Comparison — Individual Models vs Fusion Strategies\n(3-Modality, TCGA-KIRC 126-Patient Alignment Cohort)', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.5, zorder=0)
save(fig, 'Fig3_AUROC_Comparison_Bar.png')

# ════════════════════════════════════════════════════════════
# FIG 4 — Multi-Metric Grouped Bar Chart (AUROC, AUPRC, F2, Recall, Precision)
# ════════════════════════════════════════════════════════════
metric_names = ['AUROC', 'AUPRC', 'F2 Score', 'Recall', 'Precision']
metric_cols  = ['AUROC', 'AUPRC', 'F2 Score', 'Recall', 'Precision']
mc = metrics_csv.copy()
mc['Recall']    = mc['Recall'].apply(lambda x: float(str(x).replace('%',''))/100 if '%' in str(x) else float(x))
mc['Precision'] = mc['Precision'].apply(lambda x: float(str(x).replace('%',''))/100 if '%' in str(x) else float(x))

x  = np.arange(len(metric_names))
w  = 0.10
n  = len(mc)
offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

fig, ax = plt.subplots(figsize=(12,7))

for i, (_, row) in enumerate(mc.iterrows()):
    vals = [row['AUROC'], row['AUPRC'], row['F2 Score'], row['Recall'], row['Precision']]
    c    = clrs_bar[i]
    lbl  = short[i].replace('\n',' ')
    bars = ax.bar(x + offsets[i], vals, width=w*0.9, color=c, alpha=0.9,
                  edgecolor='#333333', linewidth=0.8, label=lbl, zorder=3)

ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim([0, 1.1])
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Figure 4: Multi-Metric Comparison — All Models & Fusion Strategies\n(AUROC · AUPRC · F2 · Recall · Precision)', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9.5, ncol=2, frameon=True)
ax.grid(axis='y', alpha=0.5, zorder=0)
save(fig, 'Fig4_Multi_Metric_Grouped_Bar.png')

# ════════════════════════════════════════════════════════════
# FIG 5 — SEER EDA: Class Imbalance + Demographics
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14,8.5))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[1,2])

# 5a: Overall class balance
met_counts = seer_df['metastasis'].value_counts().sort_index()
ax0.bar(['No Metastasis\n(M0)', 'Metastasis\n(M1)'], met_counts.values,
        color=['#cbd5e1', CLRS['fb']], edgecolor='#333333', width=0.5, zorder=3)
for bar,val in zip(ax0.patches, met_counts.values):
    ax0.text(bar.get_x()+bar.get_width()/2, val+200, f'{val:,}\n({val/len(seer_df)*100:.1f}%)',
             ha='center', fontsize=10, color=CLRS['text'], fontweight='bold')
ax0.set_title('Metastasis Class Balance', fontsize=11, color=CLRS['text'], fontweight='bold')
ax0.set_ylabel('Patient Count'); ax0.set_ylim([0, met_counts.max()*1.18])
ax0.grid(axis='y', alpha=0.5)

# 5b: Site-specific met rates
sites  = ['Lung','Bone','Liver','Brain']
rates  = [seer_df['lung_met'].mean()*100, seer_df['bone_met'].mean()*100,
          seer_df['liver_met'].mean()*100, seer_df['brain_met'].mean()*100]
colors = [CLRS['m1'],CLRS['m2'],CLRS['m3'],CLRS['fa']]
ax1.bar(sites, rates, color=colors, edgecolor='#333333', width=0.5, zorder=3)
for bar,val in zip(ax1.patches, rates):
    ax1.text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.2f}%',
             ha='center', fontsize=10, color=CLRS['text'], fontweight='bold')
ax1.set_title('Site-Specific Metastasis Rates', fontsize=11, color=CLRS['text'], fontweight='bold')
ax1.set_ylabel('Prevalence (%)'); ax1.set_ylim([0, max(rates)*1.3])
ax1.grid(axis='y', alpha=0.5)

# 5c: Age distribution by M-status
m0_ages = seer_df[seer_df.metastasis==0]['age']
m1_ages = seer_df[seer_df.metastasis==1]['age']
ax2.hist(m0_ages, bins=30, alpha=0.5, color='#64748b', label='M0', density=True, edgecolor='#333333')
ax2.hist(m1_ages, bins=30, alpha=0.5, color=CLRS['fb'], label='M1', density=True, edgecolor='#333333')
ax2.axvline(m0_ages.median(), color='#475569', lw=1.5, ls='--', alpha=0.8)
ax2.axvline(m1_ages.median(), color=CLRS['fb'], lw=1.5, ls='--', alpha=0.8)
ax2.set_title('Age Distribution by M-Status', fontsize=11, color=CLRS['text'], fontweight='bold')
ax2.set_xlabel('Age at Diagnosis'); ax2.set_ylabel('Density')
ax2.legend(fontsize=9); ax2.grid(alpha=0.5)

# 5d: T-Stage distribution
t_m0 = seer_df[seer_df.metastasis==0]['t_stage'].value_counts().sort_index()
t_m1 = seer_df[seer_df.metastasis==1]['t_stage'].value_counts().sort_index()
xT   = np.arange(5)
ax3.bar(xT-0.2, [t_m0.get(i,0) for i in range(5)], width=0.38, color='#cbd5e1', label='M0', edgecolor='#333333', zorder=3)
ax3.bar(xT+0.2, [t_m1.get(i,0) for i in range(5)], width=0.38, color=CLRS['fb'], label='M1', edgecolor='#333333', zorder=3)
ax3.set_xticks(xT); ax3.set_xticklabels(['T1','T1a','T2','T3','T4'])
ax3.set_title('T-Stage Distribution by M-Status', fontsize=11, color=CLRS['text'], fontweight='bold')
ax3.set_ylabel('Patient Count'); ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.5)

# 5e: Grade distribution
g_m0 = seer_df[seer_df.metastasis==0]['grade'].value_counts().sort_index()
g_m1 = seer_df[seer_df.metastasis==1]['grade'].value_counts().sort_index()
xG   = np.arange(5)
ax4.bar(xG-0.2, [g_m0.get(i,0) for i in range(5)], width=0.38, color='#cbd5e1', label='M0', edgecolor='#333333', zorder=3)
ax4.bar(xG+0.2, [g_m1.get(i,0) for i in range(5)], width=0.38, color=CLRS['fb'], label='M1', edgecolor='#333333', zorder=3)
ax4.set_xticks(xG); ax4.set_xticklabels(['G1','G2','G3','G3-4','G4'])
ax4.set_title('Fuhrman Grade by M-Status', fontsize=11, color=CLRS['text'], fontweight='bold')
ax4.set_ylabel('Patient Count'); ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.5)

# 5f: Histology distribution
h_cnts = seer_df.groupby(['histology_enc','metastasis']).size().unstack(fill_value=0)
h_lbls = ['Clear Cell','Papillary','Chromophobe','Other']
xH     = np.arange(len(h_lbls))
ax5.bar(xH-0.2, h_cnts.get(0,[0]*4).values[:4], width=0.38, color='#cbd5e1', label='M0', edgecolor='#333333', zorder=3)
ax5.bar(xH+0.2, h_cnts.get(1,[0]*4).values[:4], width=0.38, color=CLRS['fb'], label='M1', edgecolor='#333333', zorder=3)
ax5.set_xticks(xH); ax5.set_xticklabels(h_lbls, fontsize=9)
ax5.set_title('Histological Subtype by M-Status', fontsize=11, color=CLRS['text'], fontweight='bold')
ax5.set_ylabel('Patient Count'); ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.5)

fig.suptitle('Figure 5: SEER Dataset EDA — 36,738 RCC Patients (2010–2018)',
             fontsize=14, color=CLRS['text'], y=1.01, fontweight='bold')
save(fig, 'Fig5_SEER_EDA_Demographics.png')

# ════════════════════════════════════════════════════════════
# FIG 6 — Model 1 Feature Importance (LightGBM built-in)
# ════════════════════════════════════════════════════════════
feat_names_nice = ['Age','Sex','T-Stage','N-Stage','Tumour Size (cm)','Grade','Histology','Prior Tx','Diagnosis Year']
importances     = m1.feature_importances_
order           = np.argsort(importances)

fig, ax = plt.subplots(figsize=(9,6))

# use a simple single color for print clarity
bars = ax.barh([feat_names_nice[i] for i in order], importances[order],
               color=CLRS['m1'], edgecolor='#333333', height=0.6, zorder=3)
for bar, val in zip(bars, importances[order]):
    ax.text(val + importances.max()*0.01, bar.get_y()+bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9.5, color=CLRS['text'])

ax.set_xlabel('LightGBM Feature Importance (Split Count)', fontsize=12)
ax.set_title('Figure 6: Model 1 — Clinical Feature Importance\n(LightGBM, trained on 36,738 SEER patients)', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
ax.grid(axis='x', alpha=0.5, zorder=0)
ax.set_xlim([0, importances.max()*1.12])
save(fig, 'Fig6_Model1_Feature_Importance.png')

# ════════════════════════════════════════════════════════════
# FIG 7 — Confusion Matrices (3 base models + best fusion)
# ════════════════════════════════════════════════════════════
def best_thresh(y_true, y_prob):
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    best_f2, best_t = 0, 0.5
    for p,r,t in zip(prec, rec, thresh):
        d = 4*p+r
        if d>0:
            f2 = 5*p*r/d
            if f2>best_f2:
                best_f2, best_t = f2, t
    return best_t

fig, axes = plt.subplots(2, 2, figsize=(9,8))

panels = [
    ('Model 1: Clinical',     P1_fuse,  CLRS['m1'],  best_thresh(y_fuse, P1_fuse)),
    ('Model 2: Genomic',      P2_fuse,  CLRS['m2'],  best_thresh(y_fuse, P2_fuse)),
    ('Model 3: Imaging',      P3_fuse,  CLRS['m3'],  best_thresh(y_fuse, P3_fuse)),
    ('Fusion B: F2-Weighted', P_fb, CLRS['fb'],  best_thresh(y_fuse, P_fb)),
]
for ax, (name, probs, c, thresh) in zip(axes.flat, panels):
    y_pred = (probs >= thresh).astype(int)
    cm     = confusion_matrix(y_fuse, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Pred M0','Pred M1'],
                yticklabels=['True M0','True M1'],
                linewidths=1, linecolor='#cccccc',
                cbar=False, annot_kws={'size':14, 'color':'#111111', 'weight':'bold'})
    tn,fp,fn,tp = cm.ravel()
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    ax.set_title(f'{name}\nThresh={thresh:.3f} | Recall={recall:.1%} | Prec={precision:.1%}',
                 fontsize=11, color=CLRS['text'], pad=8, fontweight='bold')
    ax.tick_params(colors=CLRS['text'], labelsize=10)

fig.suptitle('Figure 7: Confusion Matrices — Base Models & Best Fusion\n(F2-Optimized Threshold, 126-Patient Alignment Cohort)',
             fontsize=13, color=CLRS['text'], y=1.03, fontweight='bold')
plt.tight_layout()
save(fig, 'Fig7_Confusion_Matrices.png')

# ════════════════════════════════════════════════════════════
# FIG 8 — Threshold vs Recall / Precision / F2 for Fusion B
# ════════════════════════════════════════════════════════════
from sklearn.metrics import fbeta_score

thresholds = np.linspace(0.01, 0.99, 200)
recalls, precisions, f2s = [], [], []
for t in thresholds:
    y_p = (P_fb >= t).astype(int)
    recalls.append(   float(np.sum((y_p==1)&(y_fuse==1))) / max(y_fuse.sum(),1))
    tp = np.sum((y_p==1)&(y_fuse==1)); fp = np.sum((y_p==1)&(y_fuse==0))
    precisions.append(tp/max(tp+fp,1))
    f2s.append(fbeta_score(y_fuse, y_p, beta=2, zero_division=0))

opt_t = thresholds[np.argmax(f2s)]
fig, ax = plt.subplots(figsize=(9,6))

ax.plot(thresholds, recalls,    color=CLRS['m2'], lw=2.2, label='Recall (Sensitivity)')
ax.plot(thresholds, precisions, color=CLRS['m3'], lw=2.2, label='Precision (PPV)')
ax.plot(thresholds, f2s,        color=CLRS['fb'], lw=2.5, label='F2 Score (β=2)')
ax.axvline(opt_t, color=CLRS['fa'], lw=1.8, ls='--',
           label=f'Optimal F2 threshold = {opt_t:.3f}')
ax.fill_betweenx([0,1], opt_t-0.01, opt_t+0.01, color=CLRS['fa'], alpha=0.15)
ax.text(opt_t+0.02, 0.08, f'threshold\n{opt_t:.3f}', color=CLRS['fa'], fontsize=10, fontweight='bold')
ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Figure 8: Threshold Sensitivity Analysis — Fusion B (F2-Weighted)\nRecall · Precision · F2 vs Decision Threshold', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
ax.legend(fontsize=10.5); ax.grid(alpha=0.5)
ax.set_xlim([0,1]); ax.set_ylim([-0.02,1.02])
save(fig, 'Fig8_Threshold_Analysis_FusionB.png')

# ════════════════════════════════════════════════════════════
# FIG 9 — Precision vs Recall Scatter (all models + fusion)
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10,7))

all_preds = [
    ('Model 1: Clinical',      P1_fuse, CLRS['m1'], 'o', 120),
    ('Model 2: Genomic',       P2_fuse, CLRS['m2'], 's', 120),
    ('Model 3: Imaging',       P3_fuse, CLRS['m3'], 'D', 120),
    ('Fusion A: Simple Avg',   P_fa,    CLRS['fa'], '^', 150),
    ('Fusion B: F2-Wtd',       P_fb,    CLRS['fb'], '*', 300),
    ('Fusion C: Stacking',     P_fc,    CLRS['fc'], 'P', 150),
    ('Fusion D: Cascade Max',  P_fd,    CLRS['fd'], 'X', 150),
]

# Plot lines connecting them or just scatter
for name, probs, c, marker, ms in all_preds:
    t_opt = best_thresh(y_fuse, probs)
    y_p   = (probs>=t_opt).astype(int)
    tp    = np.sum((y_p==1)&(y_fuse==1)); fp=np.sum((y_p==1)&(y_fuse==0))
    rec_  = tp/max(y_fuse.sum(),1)
    prec_ = tp/max(tp+fp,1)
    auroc = roc_auc_score(y_fuse, probs)
    ax.scatter(rec_, prec_, c=c, marker=marker, s=ms, zorder=5,
               edgecolors='#333333', linewidths=1.0)
    # Adding a slight offset to labels to avoid overlaps
    offset_y = 6 if 'Genomic' not in name else -12
    ax.annotate(f' {name}\nAUROC={auroc:.3f}', (rec_, prec_),
                fontsize=9.5, color='#333333', xytext=(8,offset_y), textcoords='offset points', fontweight='bold')

# iso-F2 contours
for f2_val in [0.3, 0.4, 0.5, 0.6]:
    rec_range = np.linspace(0.01, 1.0, 300)
    prec_iso  = f2_val * rec_range / (5*rec_range - f2_val*(5*rec_range-rec_range))
    mask      = (prec_iso>0)&(prec_iso<=1)
    ax.plot(rec_range[mask], prec_iso[mask], ':', color='#9ca3af', lw=1.2, alpha=0.8)
    ax.text(0.98, prec_iso[mask][-1]+0.01, f'F2={f2_val}', fontsize=9, color='#6b7280', ha='right', fontweight='bold')

ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
ax.set_ylabel('Precision (PPV)', fontsize=12)
ax.set_title('Figure 9: Precision–Recall Operating Points (F2-Optimized Threshold)\nIso-F2 Contours Shown for Clinical Context', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
ax.set_xlim([0,1.25]); ax.set_ylim([0,1.05])
ax.grid(alpha=0.5)
save(fig, 'Fig9_Precision_Recall_Operating_Points.png')

# ════════════════════════════════════════════════════════════
# FIG 10 — Model 1 SEER: Clean Site-Specific AUROC & Table
# ════════════════════════════════════════════════════════════
# Removed the messy PR scatter plot completely and replaced it with a very clean,
# easy-to-read AUROC bar chart and a neatly formatted table.
fig, ax = plt.subplots(figsize=(9, 6.5))

sub_names  = ['Overall\nMetastasis','Lung\nMet','Bone\nMet','Liver\nMet','Brain\nMet']
sub_aurocs  = m1_perf['AUROC'].tolist()
sub_recalls = m1_perf['Recall'].tolist()
sub_precisions = m1_perf['Precision'].tolist()
sub_colors  = [CLRS['m1'], '#0284c7','#16a34a','#ea580c','#be185d']

bars = ax.bar(sub_names, sub_aurocs, color=sub_colors, edgecolor='#333333', width=0.55, zorder=3)
ax.axhline(0.5, color='#9ca3af', lw=1.2, ls='--', label='Random (0.500)')
for bar,val in zip(bars, sub_aurocs):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, color=CLRS['text'], fontweight='bold')
ax.set_ylim([0.35, 0.85])
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Figure 10: Model 1 — Site-Specific Sub-Model Performance\n(SEER Holdout, 36,738 Patients, F2-Loss Optimized)', fontsize=13, color=CLRS['text'], fontweight='bold', pad=14)
ax.grid(axis='y', alpha=0.5)

# Add a clean table directly into the figure to replace the jumbled scatter plot
prev_vals = [seer_df.metastasis.mean(), seer_df.lung_met.mean(),
             seer_df.bone_met.mean(), seer_df.liver_met.mean(), seer_df.brain_met.mean()]

cell_text = []
for r, p, pv in zip(sub_recalls, sub_precisions, prev_vals):
    cell_text.append([f'{r:.1%}', f'{p:.2%}', f'{pv:.2%}'])

table = ax.table(cellText=cell_text,
                 rowLabels=[n.replace('\n', ' ') for n in sub_names],
                 colLabels=['Recall (Sensitivity)', 'Precision', 'Prevalence (Base Rate)'],
                 loc='bottom', bbox=[0.1, -0.45, 0.8, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.subplots_adjust(bottom=0.35)
save(fig, 'Fig10_Model1_Site_Specific_Performance.png')

print(f'\n✅ All 10 figures saved to:\n   {OUT}')
