import sys, os, io, tempfile, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB (for NIfTI files)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# Custom objective (required to unpickle)
# ─────────────────────────────────────────
def f2_weighted_loss(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    fn_weight, fp_weight = 4.0, 1.0
    grad = -(fn_weight * y_true * (1 - y_pred) - fp_weight * (1 - y_true) * y_pred)
    hess = (fn_weight * y_true * y_pred * (1 - y_pred) +
            fp_weight * (1 - y_true) * y_pred * (1 - y_pred))
    return grad, hess

# ─────────────────────────────────────────
# Load all models once at startup
# ─────────────────────────────────────────
print("Loading models…")
M1_FEATURES = ['age', 'sex', 't_stage', 'n_stage', 'tumor_size_cm',
                'grade', 'histology_enc', 'prior_tx', 'year_diagnosis']

model1        = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER.pkl'))
model1_lung   = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER_lung_met.pkl'))
model1_bone   = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER_bone_met.pkl'))
model1_liver  = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER_liver_met.pkl'))
model1_brain  = joblib.load(os.path.join(BASE, 'models/dataset_1/Model1_Clinical_SEER_brain_met.pkl'))

model2        = joblib.load(os.path.join(BASE, 'models/dataset_2/Model2_Genomic_TCGA.pkl'))
scaler2       = joblib.load(os.path.join(BASE, 'models/dataset_2/Model2_Scaler.pkl'))
M2_FEATURES   = joblib.load(os.path.join(BASE, 'models/dataset_2/Model2_Features.pkl'))

model3_booster = xgb.Booster()
model3_booster.load_model(os.path.join(BASE, 'models/dataset_3/Model3_Imaging_TCGA.json'))
scaler3        = joblib.load(os.path.join(BASE, 'models/dataset_3/Model3_Scaler.pkl'))
M3_FEATURES    = joblib.load(os.path.join(BASE, 'models/dataset_3/Model3_Features.pkl'))
print("All models loaded ✓")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict_model1(data):
    row = np.array([[data[f] for f in M1_FEATURES]])
    overall = sigmoid(model1.predict(row, raw_score=True))[0]
    lung    = sigmoid(model1_lung.predict(row, raw_score=True))[0]
    bone    = sigmoid(model1_bone.predict(row, raw_score=True))[0]
    liver   = sigmoid(model1_liver.predict(row, raw_score=True))[0]
    brain   = sigmoid(model1_brain.predict(row, raw_score=True))[0]
    return float(overall), float(lung), float(bone), float(liver), float(brain)

def predict_model2(data):
    row = np.array([[data.get(g, 0.0) for g in M2_FEATURES]])
    prob = model2.predict_proba(scaler2.transform(row))[0][1]
    return float(prob)

def predict_model3(data):
    row = np.array([[data.get(f, 0.0) for f in M3_FEATURES]])
    dmat = xgb.DMatrix(scaler3.transform(row), feature_names=M3_FEATURES)
    return float(model3_booster.predict(dmat)[0])

def compute_fusions(p1, p2, p3):
    w1, w2, w3 = 0.6250, 0.5743, 0.4787
    return {
        "fusion_a_simple_avg":  round((p1 + p2 + p3) / 3.0, 4),
        "fusion_b_f2_weighted": round((w1*p1 + w2*p2 + w3*p3) / (w1+w2+w3), 4),
        "fusion_d_cascade_max": round(max(p1, p2, p3), 4),
    }

def risk_label(p): return "High Risk" if p >= 0.5 else "Moderate Risk" if p >= 0.25 else "Low Risk"
def risk_class(p): return "high" if p >= 0.5 else "moderate" if p >= 0.25 else "low"


# ─────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", m2_genes=M2_FEATURES, m3_features=M3_FEATURES)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(BASE, 'webapp/static'), filename)


# ─────────────────────────────────────────
# Routes — Prediction Endpoints
# ─────────────────────────────────────────
@app.route("/predict/clinical", methods=["POST"])
def predict_clinical():
    try:
        body = request.get_json()
        data = {f: float(body[f]) if f not in ('sex','t_stage','n_stage','grade','histology_enc','prior_tx','year_diagnosis') else int(body[f]) for f in M1_FEATURES}
        overall, lung, bone, liver, brain = predict_model1(data)
        return jsonify({"overall": round(overall,4), "lung": round(lung,4), "bone": round(bone,4),
                        "liver": round(liver,4), "brain": round(brain,4),
                        "risk": risk_label(overall), "risk_class": risk_class(overall)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/genomic", methods=["POST"])
def predict_genomic():
    try:
        body = request.get_json()
        prob = predict_model2(body)
        return jsonify({"probability": round(prob,4), "risk": risk_label(prob), "risk_class": risk_class(prob)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/imaging", methods=["POST"])
def predict_imaging():
    try:
        body = request.get_json()
        prob = predict_model3(body)
        return jsonify({"probability": round(prob,4), "risk": risk_label(prob), "risk_class": risk_class(prob)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/fusion", methods=["POST"])
def predict_fusion():
    try:
        body    = request.get_json()
        clin    = body.get("clinical", {})
        genomic = body.get("genomic", {})
        imaging = body.get("imaging", {})
        p1_overall, lung, bone, liver, brain = predict_model1(clin)
        p2 = predict_model2(genomic)
        p3 = predict_model3(imaging)
        fusions = compute_fusions(p1_overall, p2, p3)
        return jsonify({
            "model1_overall": round(p1_overall,4), "model1_lung": round(lung,4),
            "model1_bone": round(bone,4), "model1_liver": round(liver,4), "model1_brain": round(brain,4),
            "model2": round(p2,4), "model3": round(p3,4), **fusions,
            "final_verdict": risk_label(fusions["fusion_b_f2_weighted"]),
            "final_risk_class": risk_class(fusions["fusion_b_f2_weighted"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────
# Routes — File Upload Endpoints
# ─────────────────────────────────────────

@app.route("/upload/genomic", methods=["POST"])
def upload_genomic():
    """
    Accept a CSV file with columns: gene, expression_value_TPM
    OR a wide-format CSV with gene names as columns and one sample row.
    Extracts the 54 required genes and runs Model 2.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files['file']
        content = f.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))

        gene_vals = {}

        # Format 1: two-column (gene, value)
        if df.shape[1] == 2:
            df.columns = [c.strip() for c in df.columns]
            gene_col = df.columns[0]
            val_col  = df.columns[1]
            for _, row in df.iterrows():
                gene_vals[str(row[gene_col]).strip()] = float(row[val_col])

        # Format 2: wide format — gene names are column headers, first row is values
        else:
            for col in df.columns:
                try:
                    gene_vals[col.strip()] = float(df[col].iloc[0])
                except:
                    pass

        # Check how many of the 54 required genes were found
        found = [g for g in M2_FEATURES if g in gene_vals]
        missing = [g for g in M2_FEATURES if g not in gene_vals]

        # Fill missing with 0
        data = {g: gene_vals.get(g, 0.0) for g in M2_FEATURES}
        prob = predict_model2(data)

        return jsonify({
            "probability":  round(prob, 4),
            "risk":         risk_label(prob),
            "risk_class":   risk_class(prob),
            "genes_found":  len(found),
            "genes_total":  len(M2_FEATURES),
            "missing_genes": missing[:10],  # show first 10 missing
            "gene_values":  {g: round(v,4) for g, v in data.items()},
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400


@app.route("/upload/radiomics/csv", methods=["POST"])
def upload_radiomics_csv():
    """
    Accept a PyRadiomics-output CSV file.
    Two supported formats:
      - Two-column (feature, value) — as produced by our sample file
      - Wide format — feature names as columns (standard PyRadiomics batch output)
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files['file']
        content = f.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content))

        feat_vals = {}

        # Format 1: two-column (feature, value)
        if df.shape[1] == 2:
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                feat_vals[str(row.iloc[0]).strip()] = float(row.iloc[1])
        # Format 2: wide format
        else:
            for col in df.columns:
                try:
                    feat_vals[col.strip()] = float(df[col].iloc[0])
                except:
                    pass

        found   = [f for f in M3_FEATURES if f in feat_vals]
        missing = [f for f in M3_FEATURES if f not in feat_vals]
        data    = {feat: feat_vals.get(feat, 0.0) for feat in M3_FEATURES}
        prob    = predict_model3(data)

        return jsonify({
            "probability":     round(prob, 4),
            "risk":            risk_label(prob),
            "risk_class":      risk_class(prob),
            "features_found":  len(found),
            "features_total":  len(M3_FEATURES),
            "missing_features": missing[:10],
            "feature_values":  {feat: round(v,6) for feat, v in data.items()},
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400


@app.route("/upload/radiomics/nifti", methods=["POST"])
def upload_radiomics_nifti():
    """
    Accept a 3D CT scan NIfTI file (.nii or .nii.gz) + tumour segmentation mask.
    Runs PyRadiomics to extract features, then runs Model 3.
    This is the real-world clinical workflow.
    """
    try:
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({"error": "Both 'image' (.nii.gz CT scan) and 'mask' (.nii.gz tumour mask) files are required"}), 400

        image_file = request.files['image']
        mask_file  = request.files['mask']

        # Save to temp directory
        tmp_dir = tempfile.mkdtemp()
        img_path  = os.path.join(tmp_dir, 'image.nii.gz')
        mask_path = os.path.join(tmp_dir, 'mask.nii.gz')

        image_file.save(img_path)
        mask_file.save(mask_path)

        # Run PyRadiomics
        from radiomics import featureextractor
        params = {
            'imageType': {'Original': {}},
            'featureClass': {
                'shape': None,
                'firstorder': None,
                'glcm': None,
                'glrlm': None,
                'glszm': None,
            },
            'setting': {
                'binWidth': 25,
                'resampledPixelSpacing': None,
                'interpolator': 'sitkBSpline',
                'verbose': False,
            }
        }
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        result = extractor.execute(img_path, mask_path)

        # Convert to dict of feature_name -> value
        feat_vals = {}
        for key, val in result.items():
            if key.startswith('original_'):
                try:
                    feat_vals[key] = float(val)
                except:
                    pass

        # Cleanup temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        found   = [f for f in M3_FEATURES if f in feat_vals]
        missing = [f for f in M3_FEATURES if f not in feat_vals]
        data    = {feat: feat_vals.get(feat, 0.0) for feat in M3_FEATURES}
        prob    = predict_model3(data)

        return jsonify({
            "probability":        round(prob, 4),
            "risk":               risk_label(prob),
            "risk_class":         risk_class(prob),
            "features_extracted": len(feat_vals),
            "features_used":      len(found),
            "missing_features":   missing[:10],
            "feature_values":     {feat: round(v,6) for feat, v in data.items()},
            "note":               f"PyRadiomics extracted {len(feat_vals)} features from CT scan. {len(found)}/{len(M3_FEATURES)} required features found."
        })
    except Exception as e:
        return jsonify({"error": f"NIfTI processing failed: {str(e)}\n{traceback.format_exc()}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
