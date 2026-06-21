const API_BASE_URL = "https://editing-lions-powers-thou.trycloudflare.com";

/* ── Theme Toggle ── */
function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  document.getElementById('theme-toggle').innerText = newTheme === 'light' ? '🌙' : '☀️';
}

document.addEventListener('DOMContentLoaded', () => {
  const savedTheme = localStorage.getItem('theme') || 'dark';
  if (savedTheme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) toggleBtn.innerText = '🌙';
  }
});
/* ── Tab switching ── */
function switchTab(tab, e) {
  if (e) e.preventDefault();
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.remove('hidden');
  if (e && e.currentTarget) e.currentTarget.classList.add('active');
  // Also highlight by tab name match
  document.querySelectorAll('.pill').forEach(p => {
    if (p.getAttribute('data-tab') === tab) p.classList.add('active');
  });
}

function activeFusionTab(id, e) {
  if (e) e.preventDefault();
  document.querySelectorAll('.f-panel').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('.ftab').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.remove('hidden');
  if (e && e.currentTarget) e.currentTarget.classList.add('active');
}

function toggleModalityTab(mod) {
  const isChecked = document.getElementById('enable_' + mod).checked;
  const tabBtn = document.getElementById('ftab-' + mod);
  if (tabBtn) {
    if (isChecked) {
      tabBtn.style.display = 'inline-block';
    } else {
      tabBtn.style.display = 'none';
      if (tabBtn.classList.contains('active')) {
        const visibleTabs = Array.from(document.querySelectorAll('.fusion-tabs .ftab')).filter(b => b.style.display !== 'none');
        if (visibleTabs.length > 0) {
          visibleTabs[0].click();
        } else {
          document.querySelectorAll('.f-panel').forEach(p => p.classList.add('hidden'));
        }
      }
    }
  }
}

/* ── Loading ── */
function showLoading() {
  const el = document.getElementById('loading');
  el.style.display = 'flex';
}
function hideLoading() {
  const el = document.getElementById('loading');
  el.style.display = 'none';
}

// Safety: hide loading overlay after max 15 seconds to avoid stuck state
document.addEventListener('DOMContentLoaded', function() {
  setTimeout(function() {
    const overlay = document.getElementById('loading');
    if (overlay && !overlay.classList.contains('hidden')) hideLoading();
  }, 15000);
});

/* ── Helpers ── */
function pctLabel(p) { return (p * 100).toFixed(1) + '%'; }
function riskCls(p) { return p >= 0.5 ? 'high' : p >= 0.25 ? 'moderate' : 'low'; }
function riskLbl(p) { return p >= 0.5 ? 'High Risk' : p >= 0.25 ? 'Moderate Risk' : 'Low Risk'; }

function probBar(label, prob, notes = '') {
  const cls = riskCls(prob);
  const pct = (prob * 100).toFixed(1);
  return `
    <div class="prob-item">
      <div class="prob-header">
        <span class="prob-label">${label}${notes ? ' <small style="color:var(--text3)">'+notes+'</small>' : ''}</span>
        <span class="prob-value ${cls}">${pct}%</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill ${cls}" style="width:${pct}%"></div>
      </div>
    </div>`;
}

/* ══════════════════════════════════════
   MODEL 1 – CLINICAL
══════════════════════════════════════ */
function fillClinicalDemo() {
  // RCC-SEER-00004: age=62, sex=Male, t_stage=T1(0), n_stage=N0(0),
  // tumor_size=3.5cm, grade=Grade1(0), histology=Other(3), prior_tx=No, year=2012
  // Ground truth: metastasis=1 (bone met), survival_months=1.0
  document.getElementById('c_age').value          = 62;
  document.getElementById('c_sex').value          = 1;
  document.getElementById('c_t_stage').value      = 0;
  document.getElementById('c_n_stage').value      = 0;
  document.getElementById('c_tumor_size').value   = 3.5;
  document.getElementById('c_grade').value        = 0;
  document.getElementById('c_histology').value    = 3;
  document.getElementById('c_prior_tx').value     = 0;
  document.getElementById('c_year').value         = 2012;
}

async function runClinical() {
  const payload = {
    age:           parseFloat(document.getElementById('c_age').value),
    sex:           parseInt(document.getElementById('c_sex').value),
    t_stage:       parseInt(document.getElementById('c_t_stage').value),
    n_stage:       parseInt(document.getElementById('c_n_stage').value),
    tumor_size_cm: parseFloat(document.getElementById('c_tumor_size').value),
    grade:         parseInt(document.getElementById('c_grade').value),
    histology_enc: parseInt(document.getElementById('c_histology').value),
    prior_tx:      parseInt(document.getElementById('c_prior_tx').value),
    year_diagnosis:parseInt(document.getElementById('c_year').value),
  };

  showLoading();
  try {
    const res  = await fetch(API_BASE_URL + '/predict/clinical', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();
    hideLoading();
    if (data.error) throw new Error(data.error);
    renderClinicalResult(data);
  } catch(e) {
    hideLoading();
    document.getElementById('result-clinical').innerHTML = `<div class="error-box">⚠ Error: ${e.message}</div>`;
    document.getElementById('result-clinical').classList.remove('hidden');
  }
}

// removed siteRiskArrow function
const lungSVG = '<svg width="1.2em" height="1.2em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:text-bottom"><path d="M12 2v6"/><path d="M8 12c-1.5-1-4-1-4 3s2.5 6 4 6 2-2 2-4V8c0-1.5 1-2.5 2-2.5"/><path d="M16 12c1.5-1 4-1 4 3s-2.5 6-4 6-2-2-2-4V8c0-1.5-1-2.5-2-2.5"/></svg>';

function renderClinicalResult(d) {
  const el = document.getElementById('result-clinical');
  el.classList.remove('hidden');

  const siteEmoji  = { lung: lungSVG, bone: '🦴', liver: '🩸', brain: '🧠' };
  const siteProbMap = { lung: d.lung, bone: d.bone, liver: d.liver, brain: d.brain };

  let sitesHTML = Object.entries(siteProbMap).map(([site, p]) => `
    <div class="site-card">
      <div class="site-icon">${siteEmoji[site]}</div>
      <div class="site-name">${site.charAt(0).toUpperCase()+site.slice(1)}</div>
      <div class="site-prob ${riskCls(p)}" style="font-size:20px; font-weight:800; margin-top:8px;">${pctLabel(p)}</div>
    </div>`).join('');

  el.innerHTML = `
    <div class="result-header">
      <div class="result-title">🏥 Clinical Model — SEER Prediction</div>
      <div class="risk-badge ${d.risk_class}">${d.risk}</div>
    </div>
    <div class="model-info-chips">
      <span class="info-chip">AUROC 0.770 on SEER Holdout</span>
      <span class="info-chip">LightGBM · F2-Weighted Loss</span>
      <span class="info-chip">36,738 Patients</span>
      <span class="info-chip">Threshold 0.496</span>
    </div>
    <div class="prob-grid">
      ${probBar('Overall Metastasis Risk Score', d.overall, '(F2-optimized)')}
    </div>
    <div class="site-section-header">
      <span>Site-Specific Risk Indices</span>
      <button class="disclaimer-toggle" onclick="this.parentElement.nextElementSibling.classList.toggle('hidden')" title="What are these indices?">ℹ️ What is this?</button>
    </div>
    <div class="site-disclaimer hidden">
      <div class="disclaimer-box">
        <strong>⚠️ Scientific Note — Relative Risk Index, Not Absolute Probability</strong>
        <p>These site-specific scores are <em>not</em> calibrated clinical probabilities. The site-specific models were trained with an asymmetric F2-loss (False Negatives penalised 4× more than False Positives) combined with SMOTE oversampling — a design that maximises sensitivity (Recall ≥ 95%) but produces uncalibrated raw sigmoid scores. On the SEER holdout, site-specific Precision is 2–4% at &gt;95% Recall, meaning the model flags almost everyone as high-risk to avoid any missed case. The index reflects <strong>relative risk rank</strong> within the population, not an absolute "X% chance of metastasis." Use the Overall Risk Score and the fused output for quantitative interpretation.</p>
      </div>
    </div>
    <div class="site-grid">${sitesHTML}</div>
    <p style="margin-top:16px; font-size:12px; color:var(--text3)">
      Model trained on 36,738 SEER patients · LightGBM · Custom F2-weighted loss · Threshold 0.496
    </p>`;
}

/* ══════════════════════════════════════
   MODEL 2 – GENOMIC
══════════════════════════════════════ */
const GENOMIC_DEMO = {
  SHCBP1: 8.4, PLEKHA9: 2.1, SPINLW1: 0.5, ZBED2: 1.8, TMEM81: 3.2,
  SLCO5A1: 0.9, TMEM220: 1.1, CDK1: 12.7, CDC20: 9.8, ITGAE: 0.3,
  PRDM8: 0.4, INHBE: 2.5, ACSS3: 3.8, FKBP15: 4.1, DRP2: 2.9,
  GOLGA8C: 1.2, PPP2R2C: 0.7, ATP7B: 1.5, IL20RB: 0.2, KIF2C: 8.9,
  SLC31A1: 5.4, WFDC10B: 0.1, RTL1: 3.3, PPIAL4G: 0.6, SSTR3: 0.8,
  CCNA1: 7.2, CCDC91: 2.0, NFE2L3: 1.3, IMPA2: 3.7, PAEP: 0.4,
  CDCA8: 10.1, LIN7A: 1.9, PITX2: 0.5, GFPT2: 4.4, UBE2S: 6.8,
  HLF: 0.3, CDCA3: 8.5, TMCC3: 1.6, TRPV3: 0.7, MOCOS: 2.3,
  HN1: 5.1, MBOAT7: 2.8, CPT2: 1.4, OR4C6: 0.2, ENPP5: 0.6,
  KRT79: 0.1, IL23A: 1.0, HSPC159: 3.5, CALR: 7.6, SOCS1: 2.2,
  ZIC2: 0.9, TNNT1: 0.4, IGF2BP3: 9.3, KIF23: 11.2
};

function fillGenomicDemo() {
  Object.entries(GENOMIC_DEMO).forEach(([gene, val]) => {
    const el = document.getElementById('g_' + gene);
    if (el) el.value = val;
  });
}

function clearGenomic() {
  document.querySelectorAll('#tab-genomic .gene-input').forEach(el => el.value = 0);
}

async function runGenomic() {
  const payload = {};
  document.querySelectorAll('#tab-genomic .gene-input').forEach(el => {
    const gene = el.id.replace('g_', '');
    payload[gene] = parseFloat(el.value) || 0;
  });

  showLoading();
  try {
    const res  = await fetch(API_BASE_URL + '/predict/genomic', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();
    hideLoading();
    if (data.error) throw new Error(data.error);
    renderGenomicResult(data);
  } catch(e) {
    hideLoading();
    document.getElementById('result-genomic').innerHTML = `<div class="error-box">⚠ Error: ${e.message}</div>`;
    document.getElementById('result-genomic').classList.remove('hidden');
  }
}

function renderGenomicResult(d) {
  const el = document.getElementById('result-genomic');
  el.classList.remove('hidden');
  el.innerHTML = `
    <div class="result-header">
      <div class="result-title">🧬 Genomic Model — TCGA RNA-Seq Prediction</div>
      <div class="risk-badge ${d.risk_class}">${d.risk} · AUROC 0.738</div>
    </div>
    <div class="prob-grid">
      ${probBar('Metastasis Probability (Genomic Signature)', d.probability, '54-gene ANOVA-selected profile')}
    </div>
    <p style="margin-top:16px; font-size:12px; color:var(--text3)">
      Model trained on 418 TCGA-KIRC RNA-Seq patients · LinearSVC + SMOTE · Recall 94.4% · Threshold 0.089
    </p>`;
}

/* ══════════════════════════════════════
   MODEL 3 – IMAGING
══════════════════════════════════════ */
const IMAGING_DEMO = {
  original_shape_Elongation: 0.72,
  original_shape_Flatness: 0.45,
  original_shape_LeastAxisLength: 48.3,
  original_shape_MajorAxisLength: 107.4,
  original_shape_Maximum2DDiameterSlice: 112.5,
  original_shape_MeshVolume: 187400.0,
  original_shape_MinorAxisLength: 77.5,
  original_shape_Sphericity: 0.62,
  original_shape_SurfaceVolumeRatio: 0.028,
  original_firstorder_10Percentile: 54.0,
  original_firstorder_90Percentile: 312.0,
  original_firstorder_Energy: 3.8e9,
  original_firstorder_Entropy: 5.8,
  original_firstorder_InterquartileRange: 178.0,
  original_firstorder_Kurtosis: 2.95,
  original_firstorder_Maximum: 485.0,
  original_firstorder_Minimum: 2.0,
  original_firstorder_Skewness: 0.45,
  original_firstorder_TotalEnergy: 3.8e9,
  original_firstorder_Uniformity: 0.038,
  original_firstorder_Variance: 6800.0,
  original_glcm_ClusterProminence: 4500000.0,
  original_glcm_ClusterShade: -280.0,
  original_glcm_Contrast: 320.0,
  original_glcm_Correlation: 0.78,
  original_glcm_DifferenceAverage: 12.4,
  original_glcm_Idmn: 0.91,
  original_glcm_Idn: 0.88,
  original_glcm_Imc1: -0.32,
  original_glcm_Imc2: 0.87,
  original_glcm_InverseVariance: 0.043,
  original_glcm_JointAverage: 182.0,
  original_glcm_MCC: 0.79,
  original_glrlm_GrayLevelNonUniformity: 28000.0,
  original_glrlm_LongRunEmphasis: 1.18,
  original_glrlm_LongRunHighGrayLevelEmphasis: 68000.0,
  original_glrlm_LongRunLowGrayLevelEmphasis: 0.00015,
  original_glrlm_RunEntropy: 4.9,
  original_glrlm_RunLengthNonUniformity: 32000.0,
  original_glszm_GrayLevelNonUniformity: 15000.0,
  original_glszm_GrayLevelNonUniformityNormalized: 0.048,
  original_glszm_GrayLevelVariance: 5900.0,
  original_glszm_LargeAreaEmphasis: 75000.0,
  original_glszm_LargeAreaHighGrayLevelEmphasis: 2.1e10,
  original_glszm_SizeZoneNonUniformity: 8800.0,
  original_glszm_SizeZoneNonUniformityNormalized: 0.28,
  original_glszm_SmallAreaLowGrayLevelEmphasis: 2.1e-7,
  original_glszm_ZoneEntropy: 8.9,
  original_glszm_ZonePercentage: 0.32,
};

function fillImagingDemo() {
  Object.entries(IMAGING_DEMO).forEach(([feat, val]) => {
    const el = document.getElementById('r_' + feat);
    if (el) el.value = val;
  });
}

function clearImaging() {
  document.querySelectorAll('#tab-imaging .gene-input').forEach(el => el.value = 0);
}

async function runImaging() {
  const payload = {};
  document.querySelectorAll('#tab-imaging .gene-input').forEach(el => {
    const feat = el.id.replace('r_', '');
    payload[feat] = parseFloat(el.value) || 0;
  });

  showLoading();
  try {
    const res  = await fetch(API_BASE_URL + '/predict/imaging', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();
    hideLoading();
    if (data.error) throw new Error(data.error);
    renderImagingResult(data);
  } catch(e) {
    hideLoading();
    document.getElementById('result-imaging').innerHTML = `<div class="error-box">⚠ Error: ${e.message}</div>`;
    document.getElementById('result-imaging').classList.remove('hidden');
  }
}

function renderImagingResult(d) {
  const el = document.getElementById('result-imaging');
  el.classList.remove('hidden');
  el.innerHTML = `
    <div class="result-header">
      <div class="result-title">☢️ Imaging Model — 3D PyRadiomics Prediction</div>
      <div class="risk-badge ${d.risk_class}">${d.risk} · AUROC 0.638</div>
    </div>
    <div class="prob-grid">
      ${probBar('Metastasis Probability (Radiomic Signature)', d.probability, '49 PyRadiomics features')}
    </div>
    <p style="margin-top:16px; font-size:12px; color:var(--text3)">
      Model trained on TCGA-KIRC preoperative CT/MRI · XGBoost + SMOTE · Recall 100% · Threshold 0.017
    </p>`;
}

/* ══════════════════════════════════════
   3-MODALITY FUSION
══════════════════════════════════════ */
function fillFusionDemo() {
  // Clinical: RCC-SEER-00004 (real patient, metastasis=1, bone_met=1)
  document.getElementById('fc_age').value        = 62;
  document.getElementById('fc_sex').value        = 1;
  document.getElementById('fc_t_stage').value    = 0;
  document.getElementById('fc_n_stage').value    = 0;
  document.getElementById('fc_tumor_size').value = 3.5;
  document.getElementById('fc_grade').value      = 0;
  document.getElementById('fc_histology').value  = 3;
  document.getElementById('fc_prior_tx').value   = 0;
  document.getElementById('fc_year').value       = 2012;

  // Genomic
  Object.entries(GENOMIC_DEMO).forEach(([gene, val]) => {
    const el = document.getElementById('fg_' + gene);
    if (el) el.value = val;
  });

  // Imaging
  Object.entries(IMAGING_DEMO).forEach(([feat, val]) => {
    const el = document.getElementById('fr_' + feat);
    if (el) el.value = val;
  });
}

async function uploadCTForAutoSegment() {
  const fileInput = document.getElementById('fusion-ct-upload');
  const statusDiv = document.getElementById('auto-segment-status');
  
  if (!fileInput.files.length) {
    alert("Please select a .nii or .nii.gz file first.");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('image', file);

  statusDiv.innerHTML = `<span style="color:var(--gold);">⏳ Uploading CT and running TotalSegmentator AI... This may take a few minutes depending on server load. Please wait...</span>`;
  showLoading();

  try {
    const res = await fetch(API_BASE_URL + '/upload/radiomics/auto-segment', {
      method: 'POST',
      body: formData
    });
    
    const data = await res.json();
    hideLoading();

    if (data.error) {
      statusDiv.innerHTML = `<span style="color:var(--error);">❌ Error: ${data.error}</span>`;
      return;
    }

    // Auto-fill the manual inputs
    if (data.feature_values) {
      for (const [feat, val] of Object.entries(data.feature_values)) {
        const input = document.getElementById('fr_' + feat);
        if (input) input.value = val;
      }
    }

    statusDiv.innerHTML = `<span style="color:var(--success);">✅ Success! ${data.note} The form below has been auto-filled. You can now run the fusion.</span>`;
    
  } catch(e) {
    hideLoading();
    statusDiv.innerHTML = `<span style="color:var(--error);">❌ Network Error: ${e.message}</span>`;
  }
}

function clearFusionResults() {
  const el = document.getElementById('result-fusion');
  if (el) {
    el.innerHTML = '';
    el.classList.add('hidden');
  }
}

async function runFusion() {
  const useClin = document.getElementById('enable_clinical').checked;
  const useGeno = document.getElementById('enable_genomic').checked;
  const useImag = document.getElementById('enable_imaging').checked;

  if (!useClin && !useGeno && !useImag) {
    alert("Please select at least one modality for fusion.");
    return;
  }

  let clinical = null;
  if (useClin) {
    clinical = {
      age:           parseFloat(document.getElementById('fc_age').value),
      sex:           parseInt(document.getElementById('fc_sex').value),
      t_stage:       parseInt(document.getElementById('fc_t_stage').value),
      n_stage:       parseInt(document.getElementById('fc_n_stage').value),
      tumor_size_cm: parseFloat(document.getElementById('fc_tumor_size').value),
      grade:         parseInt(document.getElementById('fc_grade').value),
      histology_enc: parseInt(document.getElementById('fc_histology').value),
      prior_tx:      parseInt(document.getElementById('fc_prior_tx').value),
      year_diagnosis:parseInt(document.getElementById('fc_year').value),
    };
  }

  let genomic = null;
  if (useGeno) {
    genomic = {};
    document.querySelectorAll('#f-genomic .gene-input').forEach(el => {
      genomic[el.id.replace('fg_', '')] = parseFloat(el.value) || 0;
    });
  }

  let imaging = null;
  if (useImag) {
    imaging = {};
    document.querySelectorAll('#f-imaging .gene-input').forEach(el => {
      imaging[el.id.replace('fr_', '')] = parseFloat(el.value) || 0;
    });
  }

  showLoading();
  try {
    const res  = await fetch(API_BASE_URL + '/predict/fusion', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ clinical, genomic, imaging })
    });
    const data = await res.json();
    hideLoading();
    if (data.error) throw new Error(data.error);
    renderFusionResult(data);
  } catch(e) {
    hideLoading();
    document.getElementById('result-fusion').innerHTML = `<div class="error-box">⚠ Error: ${e.message}</div>`;
    document.getElementById('result-fusion').classList.remove('hidden');
  }
}

function renderFusionResult(d) {
  const el = document.getElementById('result-fusion');
  el.classList.remove('hidden');

  el.innerHTML = `
    <div class="result-header">
      <div class="result-title">⚗️ 3-Modality Fusion — Final Verdict</div>
      <div class="risk-badge ${d.final_risk_class}">⬡ ${d.final_verdict}</div>
    </div>
    <div class="model-info-chips">
      <span class="info-chip">★ ${d.modality_count}-Modality Fusion Result</span>
      <span class="info-chip">Fusion B: Dynamically Weighted Average</span>
      <span class="info-chip">Missing-Modality Tolerant</span>
    </div>

    <div class="prob-grid">
      <h4 style="font-size:13px; color:var(--text3); margin-bottom:12px; text-transform:uppercase; letter-spacing:0.5px;">Base Model Risk Scores</h4>
      ${d.model1_overall !== null ? probBar('🏥 Model 1: Clinical (SEER)', d.model1_overall, 'AUROC 0.770') : ''}
      ${d.model2 !== null ? probBar('🧬 Model 2: Genomic (TCGA)', d.model2, 'AUROC 0.738 · Recall 94.4%') : ''}
      ${d.model3 !== null ? probBar('☢️ Model 3: Imaging (TCGA)', d.model3, 'AUROC 0.638 · Recall 100%') : ''}
    </div>

    <div class="fusion-result-grid" style="margin-top:28px;">
      <div style="grid-column:1/-1; font-size:13px; color:var(--text3); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Final Fused Output (F2-Optimized)</div>

      <div class="premium-fusion-card winner" style="grid-column:1/-1;">
        <div class="fusion-card-label" style="color:var(--gold);">Fusion Strategy B · Best AUROC 0.797</div>
        <div class="fusion-card-name" style="color:white; font-size:16px;">F2-Weighted Average</div>
        <div class="fusion-card-prob ${riskCls(d.fusion_b_f2_weighted)}" style="font-size:42px;">${pctLabel(d.fusion_b_f2_weighted)}</div>
        <div class="fusion-winner-tag" style="background:var(--gold); color:black;">★ Final Decision (Highest Recall/F2)</div>
      </div>
    </div>

    <details class="premium-details">
      <summary>
        View Other Fusion Strategies <span>▼</span>
      </summary>
      <div class="fusion-result-grid" style="margin-top:20px; padding-top:16px; border-top:1px solid var(--border);">
        <div class="fusion-card" style="box-shadow:none; border:1px solid var(--border);">
          <div class="fusion-card-label">Fusion A</div>
          <div class="fusion-card-name">Simple Average</div>
          <div class="fusion-card-prob ${riskCls(d.fusion_a_simple_avg)}">${pctLabel(d.fusion_a_simple_avg)}</div>
        </div>
        <div class="fusion-card" style="box-shadow:none; border:1px solid var(--border);">
          <div class="fusion-card-label">Fusion D</div>
          <div class="fusion-card-name">Cascade Max Pooling</div>
          <div class="fusion-card-prob ${riskCls(d.fusion_d_cascade_max)}">${pctLabel(d.fusion_d_cascade_max)}</div>
        </div>
      </div>
    </details>
    ${d.model1_overall !== null ? `
    <div style="margin-top:24px; padding-top:20px; border-top:1px solid var(--border);">
      <div class="site-section-header">
        <span>Site-Specific Risk Indices (Clinical Modality)</span>
        <button class="disclaimer-toggle" onclick="this.parentElement.nextElementSibling.classList.toggle('hidden')" title="What are these indices?">ℹ️ What is this?</button>
      </div>
      <div class="site-disclaimer hidden">
        <div class="disclaimer-box">
          <strong>⚠️ Scientific Note — Relative Risk Index, Not Absolute Probability</strong>
          <p>Site-specific scores are <em>not</em> calibrated clinical probabilities. They are relative risk indices derived from an asymmetric F2-loss model (FN penalised 4×) optimised for maximum sensitivity. Use the Fused Output above for quantitative risk interpretation.</p>
        </div>
      </div>
      <div class="site-grid">
        ${[[lungSVG,'Lung',d.model1_lung],['🦴','Bone',d.model1_bone],['🩸','Liver',d.model1_liver],['🧠','Brain',d.model1_brain]].map(([icon,name,p])=>`
          <div class="site-card">
            <div class="site-icon">${icon}</div>
            <div class="site-name">${name}</div>
            <div class="site-prob ${riskCls(p)}" style="font-size:20px; font-weight:800; margin-top:8px;">${pctLabel(p)}</div>
          </div>`).join('')}
      </div>
    </div>` : ''}
    <p style="margin-top:16px; font-size:12px; color:var(--text3)">
      Flexible dynamic fusion · Missing-modality tolerant architecture
    </p>`;
}

/* ══════════════════════════════════════
   MODE SWITCHING
══════════════════════════════════════ */
function setMode(modality, mode) {
  if (modality === 'genomic') {
    document.getElementById('geno-mode-file-panel').classList.toggle('hidden', mode !== 'file');
    document.getElementById('geno-mode-manual-panel').classList.toggle('hidden', mode !== 'manual');
    document.getElementById('geno-mode-file').classList.toggle('active', mode === 'file');
    document.getElementById('geno-mode-manual').classList.toggle('active', mode === 'manual');
  } else if (modality === 'imaging') {
    document.getElementById('img-mode-nifti-panel').classList.toggle('hidden', mode !== 'nifti');
    document.getElementById('img-mode-csv-panel').classList.toggle('hidden', mode !== 'csv');
    document.getElementById('img-mode-manual-panel').classList.toggle('hidden', mode !== 'manual');
    document.getElementById('img-mode-nifti').classList.toggle('active', mode === 'nifti');
    document.getElementById('img-mode-csv').classList.toggle('active', mode === 'csv');
    document.getElementById('img-mode-manual').classList.toggle('active', mode === 'manual');
  }
}

/* ══════════════════════════════════════
   DRAG & DROP + FILE SELECTION
══════════════════════════════════════ */
function dragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('dragover');
}

function dropFile(e, inputId, zoneId) {
  e.preventDefault();
  document.getElementById(zoneId).classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const input = document.getElementById(inputId);
  // Create DataTransfer to set files on input
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;
  handleFileSelect(input, zoneId, zoneId.replace('-dropzone', '-filename'));
}

function handleFileSelect(input, zoneId, nameDisplayId) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  const zone = document.getElementById(zoneId);
  const nameDisplay = document.getElementById(nameDisplayId);

  // Update zone visual
  zone.style.borderColor = 'var(--green)';
  zone.style.background  = 'rgba(16,185,129,0.05)';

  // Show filename
  if (nameDisplay) {
    nameDisplay.classList.remove('hidden');
    nameDisplay.innerHTML = `✅ ${file.name} <span style="color:var(--text3);margin-left:auto;">${(file.size/1024).toFixed(1)} KB</span>`;
  }
}

/* ══════════════════════════════════════
   GENOMIC FILE UPLOAD
══════════════════════════════════════ */
async function runGenomicFile() {
  const input = document.getElementById('geno-file-input');
  if (!input.files || !input.files[0]) {
    alert('Please select a gene expression CSV file first.');
    return;
  }
  const formData = new FormData();
  formData.append('file', input.files[0]);

  showLoading();
  try {
    const res  = await fetch(API_BASE_URL + '/upload/genomic', { method: 'POST', body: formData });
    const data = await res.json();
    hideLoading();
    if (data.error) throw new Error(data.error);
    renderGenomicFileResult(data);
  } catch(e) {
    hideLoading();
    document.getElementById('result-genomic').innerHTML = `<div class="error-box">⚠ ${e.message}</div>`;
    document.getElementById('result-genomic').classList.remove('hidden');
  }
}

function renderGenomicFileResult(d) {
  const el = document.getElementById('result-genomic');
  el.classList.remove('hidden');
  const coverage = Math.round((d.genes_found / d.genes_total) * 100);
  const missHTML = d.missing_genes && d.missing_genes.length
    ? `<div style="margin-top:12px;font-size:12px;color:var(--text3)">Missing genes (set to 0): ${d.missing_genes.join(', ')}${d.missing_genes.length >= 10 ? '…' : ''}</div>` : '';

  el.innerHTML = `
    <div class="result-header">
      <div class="result-title">🧬 Genomic Model — File Upload Result</div>
      <div class="risk-badge ${d.risk_class}">${d.risk} · AUROC 0.738</div>
    </div>
    <div style="display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap;">
      <div style="background:var(--bg4);border:1px solid var(--border);border-radius:var(--radius-sm);padding:12px 20px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:var(--accent)">${d.genes_found}/${d.genes_total}</div>
        <div style="font-size:12px;color:var(--text3)">Genes Matched</div>
      </div>
      <div style="background:var(--bg4);border:1px solid var(--border);border-radius:var(--radius-sm);padding:12px 20px;text-align:center;">
        <div style="font-size:22px;font-weight:800;color:${coverage>80?'#34d399':'#fbbf24'}">${coverage}%</div>
        <div style="font-size:12px;color:var(--text3)">Coverage</div>
      </div>
    </div>
    <div class="prob-grid">
      ${probBar('Metastasis Probability (Genomic Signature)', d.probability, '54-gene ANOVA-selected profile')}
    </div>
    ${missHTML}
    <p style="margin-top:16px;font-size:12px;color:var(--text3)">
      LinearSVC + SMOTE · Recall 94.4% · Threshold 0.089
    </p>`;
}

/* ══════════════════════════════════════
   IMAGING — NIFTI UPLOAD
══════════════════════════════════════ */
async function runImagingNifti() {
  const ctInput   = document.getElementById('ct-file-input');
  const maskInput = document.getElementById('mask-file-input');
  if (!ctInput.files || !ctInput.files[0]) { alert('Please upload a CT scan NIfTI file.'); return; }
  if (!maskInput.files || !maskInput.files[0]) { alert('Please upload a tumour mask NIfTI file.'); return; }

  const formData = new FormData();
  formData.append('image', ctInput.files[0]);
  formData.append('mask',  maskInput.files[0]);

  showLoading();
  document.querySelector('#loading p').textContent = 'Running PyRadiomics feature extraction… (10–60s)';
  try {
    const res  = await fetch(API_BASE_URL + '/upload/radiomics/nifti', { method: 'POST', body: formData });
    const data = await res.json();
    hideLoading();
    document.querySelector('#loading p').textContent = 'Running AI inference…';
    if (data.error) throw new Error(data.error);
    renderImagingFileResult(data, 'CT Scan + PyRadiomics', data.note);
  } catch(e) {
    hideLoading();
    document.querySelector('#loading p').textContent = 'Running AI inference…';
    document.getElementById('result-imaging').innerHTML = `<div class="error-box">⚠ ${e.message}</div>`;
    document.getElementById('result-imaging').classList.remove('hidden');
  }
}

/* ══════════════════════════════════════
   IMAGING — CSV UPLOAD
══════════════════════════════════════ */
async function runImagingCSV() {
  const input = document.getElementById('rad-file-input');
  if (!input.files || !input.files[0]) { alert('Please select a PyRadiomics CSV file first.'); return; }

  const formData = new FormData();
  formData.append('file', input.files[0]);

  showLoading();
  try {
    const res  = await fetch(API_BASE_URL + '/upload/radiomics/csv', { method: 'POST', body: formData });
    const data = await res.json();
    hideLoading();
    if (data.error) throw new Error(data.error);
    renderImagingFileResult(data, 'PyRadiomics CSV', `${data.features_found}/${data.features_total} features matched`);
  } catch(e) {
    hideLoading();
    document.getElementById('result-imaging').innerHTML = `<div class="error-box">⚠ ${e.message}</div>`;
    document.getElementById('result-imaging').classList.remove('hidden');
  }
}

function renderImagingFileResult(d, source, note) {
  const el = document.getElementById('result-imaging');
  el.classList.remove('hidden');
  el.innerHTML = `
    <div class="result-header">
      <div class="result-title">☢️ Imaging Model — ${source}</div>
      <div class="risk-badge ${d.risk_class}">${d.risk} · AUROC 0.638</div>
    </div>
    <div class="prob-grid">
      ${probBar('Metastasis Probability (Radiomic Signature)', d.probability, '49 PyRadiomics features')}
    </div>
    ${note ? `<div style="margin-top:12px;font-size:12px;color:var(--text3)">ℹ️ ${note}</div>` : ''}
    <p style="margin-top:16px;font-size:12px;color:var(--text3)">
      XGBoost + SMOTE · Recall 100% · Threshold 0.017
    </p>`;
}
