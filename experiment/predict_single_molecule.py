# ================================================================
# ECOAI DUAL-INFERENCE ENGINE (JUPYTER NOTEBOOK FORMAT)
# ================================================================
# A step-by-step pipeline to run simultaneous inference for
# Repellency AND Insecticidal Activity on a single SMILES string.
#
# UPDATED WITH OOD DETECTION & RELIABILITY SCORING
# ================================================================


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit import RDLogger

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT     = Path(r"G:\research\ECOAI")
DATA_DIR         = PROJECT_ROOT / "Datasets" / "data"

# Phase 4 (Repellency) Paths
REPEL_MODEL_DIR  = PROJECT_ROOT / "experiment" / "phase4_model_training" / "models"
REPEL_OOF_FILE   = DATA_DIR / "model_predictions_v1.parquet"
REPEL_FEAT_FILE  = DATA_DIR / "features_combined.parquet"

# Phase 10 (Insecticidal) Paths
INSECT_ARTIFACTS = DATA_DIR / "phase10_artifacts"
INSECT_MODEL_DIR = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation" / "models"
INSECT_OOF_FILE  = INSECT_ARTIFACTS / "classical_oof_preds.parquet"
INSECT_META_FILE = INSECT_ARTIFACTS / "insecticide_meta.parquet"

# ── Feature Configurations ─────────────────────────────────
MORGAN_RADIUS = 2
MORGAN_NBITS  = 2048

RDKIT_2D_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt), ("ExactMolWt", Descriptors.ExactMolWt),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount), ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors), ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings), ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("RingCount", Descriptors.RingCount), ("TPSA", Descriptors.TPSA),
    ("MolLogP", Descriptors.MolLogP), ("MolMR", Descriptors.MolMR),
    ("FractionCSP3", Descriptors.FractionCSP3), ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons), ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge), ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge),
    ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge), ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT), ("Chi0", Descriptors.Chi0),
    ("Chi0n", Descriptors.Chi0n), ("Chi0v", Descriptors.Chi0v),
    ("Chi1", Descriptors.Chi1), ("Chi1n", Descriptors.Chi1n),
    ("Chi1v", Descriptors.Chi1v), ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("Kappa1", Descriptors.Kappa1), ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3), ("LabuteASA", Descriptors.LabuteASA),
    ("PEOE_VSA1", Descriptors.PEOE_VSA1), ("PEOE_VSA2", Descriptors.PEOE_VSA2),
    ("PEOE_VSA3", Descriptors.PEOE_VSA3), ("PEOE_VSA6", Descriptors.PEOE_VSA6),
    ("PEOE_VSA7", Descriptors.PEOE_VSA7), ("PEOE_VSA8", Descriptors.PEOE_VSA8),
    ("SMR_VSA1", Descriptors.SMR_VSA1), ("SMR_VSA5", Descriptors.SMR_VSA5),
    ("SMR_VSA7", Descriptors.SMR_VSA7), ("SlogP_VSA2", Descriptors.SlogP_VSA2),
    ("SlogP_VSA3", Descriptors.SlogP_VSA3), ("SlogP_VSA5", Descriptors.SlogP_VSA5),
    ("EState_VSA1", Descriptors.EState_VSA1), ("EState_VSA2", Descriptors.EState_VSA2),
    ("VSA_EState1", Descriptors.VSA_EState1), ("VSA_EState2", Descriptors.VSA_EState2),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings), ("NumAromaticHeterocycles", Descriptors.NumAromaticHeterocycles),
    ("NumSaturatedHeterocycles", Descriptors.NumSaturatedHeterocycles), ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount), ("NumHeteroatoms", Descriptors.NumHeteroatoms),
]

print("[INFO] Project Configuration loaded.")


# ================================================================
# Cell 2 — Load Dual-Models into Memory
# ================================================================
models_repel = []
for fold in range(5):
    path = REPEL_MODEL_DIR / f"lgb_fold_{fold}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            models_repel.append(pickle.load(f))

models_insect = []
for fold in range(5):
    path = INSECT_MODEL_DIR / f"classical_fold_{fold}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            models_insect.append(pickle.load(f))

print(f"[INFO] Loaded Repellency Models: {len(models_repel)}")
print(f"[INFO] Loaded Insecticidal Models: {len(models_insect)}")


# ================================================================
# Cell 3 — Fit Dual Calibrators & OOD Detector
# ================================================================
# We fit this on-the-fly to keep the script small and self-contained

# ───── 1. Repellency Calibrator ─────
df_repel = pd.read_parquet(REPEL_OOF_FILE)
y_true_r = df_repel["repellent_active"].values.astype(int)
y_raw_r  = df_repel["p_repellent_winner_raw"].values

iso_repel = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso_repel.fit(y_raw_r, y_true_r)

residuals_r = np.abs(y_true_r - iso_repel.transform(y_raw_r))
q_hat_r = np.quantile(residuals_r, 0.90)

# ───── 2. Insecticidal Calibrator ─────
df_ins_oof = pd.read_parquet(INSECT_OOF_FILE)
df_ins_meta = pd.read_parquet(INSECT_META_FILE)
df_ins_merge = pd.merge(df_ins_oof, df_ins_meta, on="compound_id", how="inner")

y_true_i = df_ins_merge["insecticidal_active"].values.astype(int)
y_raw_i  = df_ins_merge["p_insecticidal_raw"].values

iso_insect = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso_insect.fit(y_raw_i, y_true_i)

residuals_i = np.abs(y_true_i - iso_insect.transform(y_raw_i))
q_hat_i = np.quantile(residuals_i, 0.90)

# ───── 3. OOD Estimator ───── 
# (Fit on merged training features to capture general chemical space)
df_train = pd.read_parquet(REPEL_FEAT_FILE)
df_train_labeled = df_train[df_train["repellent_active"].notna()].copy()
meta_cols = ["compound_id", "canonical_smiles", "source_dataset", "repellent_active", "scaffold_smiles", "fold_id", "qc_status"]
feature_cols = [c for c in df_train_labeled.columns if c not in meta_cols]

X_train = df_train_labeled[feature_cols].values
ood_imp = SimpleImputer(strategy="median").fit(X_train)
X_train_imp = ood_imp.transform(X_train)
ood_sc = StandardScaler().fit(X_train_imp)
X_train_std = ood_sc.transform(X_train_imp)

ood_nn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1)
ood_nn.fit(X_train_std)

train_dist, _ = ood_nn.kneighbors(X_train_std)
train_mean_dist = train_dist.mean()

print(f"[INFO] Dual Calibration and OOD engines ready.")


# ================================================================
# Cell 4 — Define Inference Engine
# ================================================================

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS)
    arr_morgan = np.zeros(MORGAN_NBITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr_morgan)
    arr_rdkit = []
    for _, func in RDKIT_2D_DESCRIPTORS:
        try:
            val = func(mol)
            arr_rdkit.append(np.nan if np.isinf(val) else val)
        except:
            arr_rdkit.append(np.nan)
    return np.concatenate([arr_morgan, arr_rdkit]).reshape(1, -1)

def predict_molecule(smiles):
    feat = extract_features(smiles)
    if feat is None:
        print("[ERROR] Invalid SMILES")
        return

    # ── A. OOD Check ──
    f_imp = ood_imp.transform(feat)
    f_std = ood_sc.transform(f_imp)
    dist, _ = ood_nn.kneighbors(f_std)
    ood_score = dist.mean() / train_mean_dist

    # ── B. Repellency Ensemble ──
    p_rep_raw = []
    for m in models_repel:
        x = m["scaler"].transform(m["imputer"].transform(feat))
        p_rep_raw.append(m["model"].predict_proba(x)[0, 1])
    
    raw_r_mean = np.mean(p_rep_raw)
    p_repel = float(iso_repel.transform([raw_r_mean])[0])
    p_non_repel = 1.0 - p_repel
    r_lower = max(0.0, p_repel - q_hat_r)
    r_upper = min(1.0, p_repel + q_hat_r)

    # ── C. Insecticidal Ensemble ──
    p_ins_raw = []
    for m in models_insect:
        x = m["scaler"].transform(m["imputer"].transform(feat))
        p_ins_raw.append(m["model"].predict_proba(x)[0, 1])
    
    raw_i_mean = np.mean(p_ins_raw)
    p_insect = float(iso_insect.transform([raw_i_mean])[0])
    p_non_insect = 1.0 - p_insect
    i_lower = max(0.0, p_insect - q_hat_i)
    i_upper = min(1.0, p_insect + q_hat_i)

    # ── D. Classification & Reporting ──
    print("\n" + "="*70)
    print(" ECOAI — DUAL-ACTIVITY PREDICTION REPORT (V3)")
    print("="*70)
    print(f" Molecule: {smiles}")
    print(f" OOD Score: {ood_score:.2f} " + ("(⚠️ HIGH UNCERTAINTY)" if ood_score > 2.0 else "(Reliable)"))
    print("-" * 70)
    print(" 🦟 REPELLENCY ACTIVITY")
    print(f"   > Repellent:       {p_repel*100:5.1f} %   [90% Confidence: {r_lower*100:4.1f}% - {r_upper*100:4.1f}%]")
    print(f"   > Non-Repellent:   {p_non_repel*100:5.1f} %")
    print("-" * 70)
    print(" 💀 INSECTICIDAL ACTIVITY")
    print(f"   > Insecticidal:    {p_insect*100:5.1f} %   [90% Confidence: {i_lower*100:4.1f}% - {i_upper*100:4.1f}%]")
    print(f"   > Non-Insecticidal:{p_non_insect*100:5.1f} %")
    print("-" * 70)
    
    # Simple Heuristic Classification
    if p_repel > 0.5 and p_insect < 0.2:
        category = "🟢 HOLY GRAIL (Safe Repellent)"
    elif p_repel > 0.5 and p_insect >= 0.2:
        category = "🟡 TOXIC REPELLENT (Handles insects but likely hazardous)"
    elif p_repel <= 0.5 and p_insect >= 0.5:
        category = "🔴 BUG SPRAY (Toxic Insecticide, not a repellent)"
    else:
        category = "⚪ INACTIVE (Useless)"
        
    print(f" 🎯 FINAL CLASSIFICATION: {category}")
    print("="*70 + "\n")

print("[INFO] V3 Dual Inference Engine ready.")


# ================================================================
# Cell 5 — Predict!
# ================================================================

# Try a known active compound or a random steroid
smiles_list = [
    "CC(C)N1CC[C@@]2(C[C@H]3CC[C@@H](C2)N3)OC1=O",  # Expected: Repellent
    "c1ccccc1", # Benzene (Expected: Inactive/Toxic)
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C" # Steroid
]

for s in smiles_list:
    predict_molecule(s)
