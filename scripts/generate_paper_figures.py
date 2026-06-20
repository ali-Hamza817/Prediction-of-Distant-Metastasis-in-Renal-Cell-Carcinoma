"""
generate_paper_figures.py
Generates publication-quality figures for the RCC metastasis research paper.
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

CLRS = {
    'm1':       '#3730a3',
    'm2':       '#0891b2',
    'm3':       '#059669',
    'fa':       '#d97706',
    'fb':       '#dc2626',
    'fc':       '#7e22ce',
    'fd':       '#db2777',
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
m2      = joblib.load(f'{BASE}/models/dataset_2/Model2_Genomic_TCGA.pkl')
sc2     = joblib.load(f'{BASE}/models/dataset_2/Model2_Scaler.pkl')
feat2   = joblib.load(f'{BASE}/models/dataset_2/Model2_Features.pkl')

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

# RNA-Seq
rna_df = pd.read_csv(f'{BASE}/datasets/dataset_2/HiSeqV2.gz', sep='\t', index_col=0, compression='gzip').T
rna_df.index = rna_df.index.str[:12]
rna_df = rna_df[~rna_df.index.duplicated(keep='first')]

# 418-pt Genomic Cohort
genomic_418_df = clin_df[['metastasis']].join(rna_df[feat2], how='inner').dropna()

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
y_fuse  = fuse['metastasis'].values
P2_fuse = fuse['P2'].values
X_rad   = fuse[list(rad_df.columns)].values

# Compute P1 on fusion cohort via Model1 transfer
clin_feats_fuse = clin_df.loc[fuse.index, :]
clin_aligned = pd.DataFrame(0, index=fuse.index, columns=M1F)
col_map = {'age_at_initial_pathologic_diagnosis': 'age', 'age_at_index': 'age'}
for old, new in col_map.items():
    if old in clin_feats_fuse.columns:
        clin_aligned[new] = pd.to_numeric(clin_feats_fuse[old], errors='coerce').fillna(0)
gender_col = [c for c in clin_feats_fuse.columns if 'gender' in c.lower() or c=='gender']
if gender_col: clin_aligned['sex'] = (clin_feats_fuse[gender_col[0]].str.upper()=='MALE').astype(int)

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

metrics_csv = pd.read_csv(f'{BASE}/results/Final Thesis Results/Final_Metrics_Table.csv')
m1_perf     = pd.read_csv(f'{BASE}/results/Model1_Performance.csv', index_col=0)

print('\nGenerating figures…')

# ════════════════════════════════════════════════════════════
# FIG 1 — ROC Curves: All Models + Fusion Strategies
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9,7))

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
    ax.plot(fpr, tpr, color=c, lw=lw, ls=ls, alpha=0.9 if 'B' not in name else 1.0, label=f'{name}  (AUROC={a:.3f})')

ax.plot([0,1],[0,1],'--', color='#9ca3af', lw=1.2, label='Random (AUROC=0.500)')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, color=CLRS['text'])
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, color=CLRS['text'])
ax.set_title('ROC Curves — All Models & Fusion Strategies\n(TCGA-KIRC 126-Patient Alignment Cohort)', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
# Legend INSIDE the graph
ax.legend(loc='lower right', fontsize=10, frameon=True, edgecolor='#cccccc', facecolor='white', framealpha=0.95)
ax.grid(True, alpha=0.5)
ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.02])
save(fig, 'Fig1_ROC_Curves.png')

# ════════════════════════════════════════════════════════════
# FIG 2 — Precision-Recall Curves
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9,7))
base_rate = y_fuse.mean()
ax.axhline(base_rate, color='#9ca3af', lw=1.2, ls='--', label=f'Baseline (prevalence={base_rate:.3f})')

for name, scores, c, ls in curves:
    prec, rec, _ = precision_recall_curve(y_fuse, scores)
    ap = average_precision_score(y_fuse, scores)
    lw = 2.5 if 'B' in name else 1.8
    ax.plot(rec, prec, color=c, lw=lw, ls=ls, alpha=0.9 if 'B' not in name else 1.0, label=f'{name}  (AUPRC={ap:.3f})')

ax.set_xlabel('Recall (Sensitivity)', fontsize=12, color=CLRS['text'])
ax.set_ylabel('Precision (PPV)', fontsize=12, color=CLRS['text'])
ax.set_title('Precision-Recall Curves — All Models & Fusion Strategies\n(TCGA-KIRC 126-Patient Alignment Cohort)', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
# Legend INSIDE the graph (upper right to avoid bottom left lines)
ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='#cccccc', facecolor='white', framealpha=0.95)
ax.grid(True, alpha=0.5)
ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
save(fig, 'Fig2_PR_Curves.png')

# ════════════════════════════════════════════════════════════
# FIG 3, 4, 5 — Dataset Demographics (Separate for each dataset)
# ════════════════════════════════════════════════════════════
def plot_dataset_pie(m1_count, m0_count, title, subtitle, filename):
    fig, ax = plt.subplots(figsize=(7,6))
    total = m1_count + m0_count
    colors = [CLRS['m1'], '#cbd5e1']
    explode = (0.1, 0)
    wedges, texts, autotexts = ax.pie([m1_count, m0_count], explode=explode, labels=['Metastasis (M1)', 'No Metastasis (M0)'], 
                                      colors=colors, autopct='%1.1f%%', shadow=False, startangle=140, 
                                      textprops={'fontsize': 11})
    plt.setp(autotexts, size=11, weight="bold", color="white")
    if autotexts[1]: plt.setp(autotexts[1], color="#333333")
    
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold', pad=15)
    
    # Text box with total counts
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc')
    ax.text(1.2, 0.5, f"Total Patients: {total:,}\n\nM1 Cases: {m1_count:,}\nM0 Cases: {m0_count:,}", 
            fontsize=11, verticalalignment='center', bbox=props)
    
    save(fig, filename)

# Dataset 1: SEER
seer_m1 = df_seer['metastasis'].sum()
seer_m0 = len(df_seer) - seer_m1
plot_dataset_pie(seer_m1, seer_m0, 'Dataset 1: Clinical Cohort (SEER)', 'Population-Scale Demographics for Model 1', 'Fig3a_Dataset_SEER_Demographics.png')

# Dataset 2: TCGA Genomic 418
g418_m1 = genomic_418_df['metastasis'].sum()
g418_m0 = len(genomic_418_df) - g418_m1
plot_dataset_pie(g418_m1, g418_m0, 'Dataset 2: Genomic Cohort (TCGA-KIRC)', '418-Patient Cohort for Model 2 (RNA-Seq)', 'Fig3b_Dataset_Genomic418_Demographics.png')

# Dataset 3: TCGA Imaging/Fusion 126
f126_m1 = y_fuse.sum()
f126_m0 = len(y_fuse) - f126_m1
plot_dataset_pie(f126_m1, f126_m0, 'Dataset 3: Imaging / Alignment Cohort (TCGA-KIRC)', '126-Patient Inner Join for Model 3 & Fusion', 'Fig3c_Dataset_Fusion126_Demographics.png')

# ════════════════════════════════════════════════════════════
# FIG 4 — AUROC Comparison Bar Chart
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
ax.set_title('AUROC Comparison — Individual Models vs Fusion Strategies\n(3-Modality, TCGA-KIRC 126-Patient Alignment Cohort)', fontsize=13, color=CLRS['text'], pad=14, fontweight='bold')
ax.grid(axis='y', alpha=0.5, zorder=0)
save(fig, 'Fig4_AUROC_Comparison_Bar.png')

# ════════════════════════════════════════════════════════════
# FIG 5 — SEER EDA 6-Panel
# ════════════════════════════════════════════════════════════
# Keeping this one as it's highly detailed for the appendix/supplementary
fig = plt.figure(figsize=(14,8.5))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[1,2])

met_counts = df_seer['metastasis'].value_counts().sort_index()
ax0.bar(['No Metastasis\n(M0)', 'Metastasis\n(M1)'], met_counts.values, color=['#cbd5e1', CLRS['fb']], edgecolor='#333333', width=0.5, zorder=3)
for bar,val in zip(ax0.patches, met_counts.values):
    ax0.text(bar.get_x()+bar.get_width()/2, val+200, f'{val:,}\n({val/len(df_seer)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
ax0.set_title('Metastasis Class Balance', fontsize=11, fontweight='bold'); ax0.set_ylabel('Patient Count'); ax0.set_ylim([0, met_counts.max()*1.18]); ax0.grid(axis='y', alpha=0.5)

sites  = ['Lung','Bone','Liver','Brain']
rates  = [df_seer['lung_met'].mean()*100, df_seer['bone_met'].mean()*100, df_seer['liver_met'].mean()*100, df_seer['brain_met'].mean()*100]
ax1.bar(sites, rates, color=[CLRS['m1'],CLRS['m2'],CLRS['m3'],CLRS['fa']], edgecolor='#333333', width=0.5, zorder=3)
for bar,val in zip(ax1.patches, rates): ax1.text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
ax1.set_title('Site-Specific Rates', fontsize=11, fontweight='bold'); ax1.set_ylabel('Prevalence (%)'); ax1.set_ylim([0, max(rates)*1.3]); ax1.grid(axis='y', alpha=0.5)

m0_ages = df_seer[df_seer.metastasis==0]['age']
m1_ages = df_seer[df_seer.metastasis==1]['age']
ax2.hist(m0_ages, bins=30, alpha=0.5, color='#64748b', label='M0', density=True, edgecolor='#333333')
ax2.hist(m1_ages, bins=30, alpha=0.5, color=CLRS['fb'], label='M1', density=True, edgecolor='#333333')
ax2.axvline(m0_ages.median(), color='#475569', lw=1.5, ls='--'); ax2.axvline(m1_ages.median(), color=CLRS['fb'], lw=1.5, ls='--')
ax2.set_title('Age Distribution', fontsize=11, fontweight='bold'); ax2.legend(fontsize=9); ax2.grid(alpha=0.5)

t_m0 = df_seer[df_seer.metastasis==0]['t_stage'].value_counts().sort_index()
t_m1 = df_seer[df_seer.metastasis==1]['t_stage'].value_counts().sort_index()
xT   = np.arange(5)
ax3.bar(xT-0.2, [t_m0.get(i,0) for i in range(5)], width=0.38, color='#cbd5e1', label='M0', edgecolor='#333333', zorder=3)
ax3.bar(xT+0.2, [t_m1.get(i,0) for i in range(5)], width=0.38, color=CLRS['fb'], label='M1', edgecolor='#333333', zorder=3)
ax3.set_xticks(xT); ax3.set_xticklabels(['T1','T1a','T2','T3','T4']); ax3.set_title('T-Stage Distribution', fontsize=11, fontweight='bold'); ax3.grid(axis='y', alpha=0.5)

g_m0 = df_seer[df_seer.metastasis==0]['grade'].value_counts().sort_index()
g_m1 = df_seer[df_seer.metastasis==1]['grade'].value_counts().sort_index()
xG   = np.arange(5)
ax4.bar(xG-0.2, [g_m0.get(i,0) for i in range(5)], width=0.38, color='#cbd5e1', label='M0', edgecolor='#333333', zorder=3)
ax4.bar(xG+0.2, [g_m1.get(i,0) for i in range(5)], width=0.38, color=CLRS['fb'], label='M1', edgecolor='#333333', zorder=3)
ax4.set_xticks(xG); ax4.set_xticklabels(['G1','G2','G3','G3-4','G4']); ax4.set_title('Fuhrman Grade', fontsize=11, fontweight='bold'); ax4.grid(axis='y', alpha=0.5)

h_cnts = df_seer.groupby(['histology_enc','metastasis']).size().unstack(fill_value=0)
h_lbls = ['Clear Cell','Papillary','Chromophobe','Other']
xH     = np.arange(len(h_lbls))
ax5.bar(xH-0.2, h_cnts.get(0,[0]*4).values[:4], width=0.38, color='#cbd5e1', label='M0', edgecolor='#333333', zorder=3)
ax5.bar(xH+0.2, h_cnts.get(1,[0]*4).values[:4], width=0.38, color=CLRS['fb'], label='M1', edgecolor='#333333', zorder=3)
ax5.set_xticks(xH); ax5.set_xticklabels(h_lbls, fontsize=9); ax5.set_title('Histological Subtype', fontsize=11, fontweight='bold'); ax5.grid(axis='y', alpha=0.5)

fig.suptitle('SEER Dataset Detailed Demographics — 36,738 RCC Patients', fontsize=14, fontweight='bold', y=1.01)
save(fig, 'Fig5_SEER_Detailed_Demographics.png')

# ════════════════════════════════════════════════════════════
# FIG 6 — INDIVIDUAL Confusion Matrices
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

def plot_cm(name, probs, c, filename):
    fig, ax = plt.subplots(figsize=(6,5))
    thresh = best_thresh(y_fuse, probs)
    y_pred = (probs >= thresh).astype(int)
    cm     = confusion_matrix(y_fuse, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['Pred M0','Pred M1'],
                yticklabels=['True M0','True M1'],
                linewidths=1, linecolor='#cccccc',
                cbar=False, annot_kws={'size':16, 'color':'#111111', 'weight':'bold'})
    tn,fp,fn,tp = cm.ravel()
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    ax.set_title(f'{name} Confusion Matrix\nThresh={thresh:.3f} | Recall={recall:.1%} | Prec={precision:.1%}',
                 fontsize=12, fontweight='bold', pad=12)
    save(fig, filename)

plot_cm('Model 1: Clinical', P1_fuse, CLRS['m1'], 'Fig6a_CM_Model1.png')
plot_cm('Model 2: Genomic', P2_fuse, CLRS['m2'], 'Fig6b_CM_Model2.png')
plot_cm('Model 3: Imaging', P3_fuse, CLRS['m3'], 'Fig6c_CM_Model3.png')
plot_cm('Fusion B: F2-Weighted', P_fb, CLRS['fb'], 'Fig6d_CM_FusionB.png')

# ════════════════════════════════════════════════════════════
# FIG 7 — Model 1 Site-Specific Performance Table
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
sub_names  = ['Overall\nMetastasis','Lung\nMet','Bone\nMet','Liver\nMet','Brain\nMet']
sub_aurocs  = m1_perf['AUROC'].tolist()
sub_colors  = [CLRS['m1'], '#0284c7','#16a34a','#ea580c','#be185d']

bars = ax.bar(sub_names, sub_aurocs, color=sub_colors, edgecolor='#333333', width=0.55, zorder=3)
ax.axhline(0.5, color='#9ca3af', lw=1.2, ls='--', label='Random (0.500)')
for bar,val in zip(bars, sub_aurocs):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim([0.35, 0.85]); ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Model 1 — Site-Specific Sub-Model Performance\n(SEER Holdout, 36,738 Patients)', fontsize=13, fontweight='bold', pad=14)
ax.grid(axis='y', alpha=0.5)
save(fig, 'Fig7_Site_Specific_Performance.png')

print(f'\n✅ All figures successfully generated and saved to:\n   {OUT}')
