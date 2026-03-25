import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit import RDLogger

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)

# ----------------- UI Config -----------------
st.set_page_config(page_title="ECOAI Predictor", layout="centered")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { 
        background-color: #f4f6f9; 
    }
    .block-container { 
        padding-top: 3rem; 
        padding-bottom: 3rem; 
        max-width: 750px; 
        background-color: #ffffff; 
        border-radius: 12px; 
        box-shadow: 0 8px 24px rgba(0,0,0,0.04); 
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .main-title { 
        font-size: 2.4rem; 
        font-weight: 800; 
        text-align: center; 
        color: #1a202c; 
        margin-bottom: 40px; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .alert-holy-grail { background-color: #f0fff4; border: 1px solid #c6f6d5; border-left: 6px solid #38a169; padding: 20px; border-radius: 6px; color: #22543d; }
    .alert-toxic { background-color: #fffff0; border: 1px solid #fefcbf; border-left: 6px solid #d69e2e; padding: 20px; border-radius: 6px; color: #744210; }
    .alert-bug-spray { background-color: #fff5f5; border: 1px solid #fed7d7; border-left: 6px solid #e53e3e; padding: 20px; border-radius: 6px; color: #742a2a; }
    .alert-inactive { background-color: #f7fafc; border: 1px solid #e2e8f0; border-left: 6px solid #718096; padding: 20px; border-radius: 6px; color: #2d3748; }
    
    /* Button Customization */
    .stButton>button { 
        width: 100%; 
        font-weight: 600; 
        background-color: #2b6cb0; 
        border-radius: 6px; 
        padding: 0.6rem; 
        color: white; 
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover { 
        background-color: #2c5282; 
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Caching Backend -----------------
@st.cache_resource(show_spinner=False)
def load_ecosystem():
    """Builds and caches the entire multi-task network (Models + Calibrators + OOD)."""
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "Datasets" / "data"

    REPEL_MODEL_DIR = PROJECT_ROOT / "experiment" / "phase4_model_training" / "models"
    REPEL_OOF_FILE = DATA_DIR / "model_predictions_v1.parquet"
    REPEL_FEAT_FILE = DATA_DIR / "features_combined.parquet"

    INSECT_ARTIFACTS = DATA_DIR / "phase10_artifacts"
    INSECT_MODEL_DIR = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation" / "models"
    INSECT_OOF_FILE = INSECT_ARTIFACTS / "classical_oof_preds.parquet"
    INSECT_META_FILE = INSECT_ARTIFACTS / "insecticide_meta.parquet"

    models_repel = []
    for fold in range(5):
        path = REPEL_MODEL_DIR / f"lgb_fold_{fold}.pkl"
        with open(path, "rb") as f:
            models_repel.append(pickle.load(f))

    models_insect = []
    for fold in range(5):
        path = INSECT_MODEL_DIR / f"classical_fold_{fold}.pkl"
        with open(path, "rb") as f:
            models_insect.append(pickle.load(f))

    # Repellency Calibrator
    df_repel = pd.read_parquet(REPEL_OOF_FILE)
    iso_repel = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_repel.fit(df_repel["p_repellent_winner_raw"].values, df_repel["repellent_active"].values.astype(int))
    q_hat_r = np.quantile(np.abs(df_repel["repellent_active"].values.astype(int) - iso_repel.transform(df_repel["p_repellent_winner_raw"].values)), 0.90)

    # Insecticidal Calibrator
    df_ins_merge = pd.merge(pd.read_parquet(INSECT_OOF_FILE), pd.read_parquet(INSECT_META_FILE), on="compound_id", how="inner")
    iso_insect = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_insect.fit(df_ins_merge["p_insecticidal_raw"].values, df_ins_merge["insecticidal_active"].values.astype(int))
    q_hat_i = np.quantile(np.abs(df_ins_merge["insecticidal_active"].values.astype(int) - iso_insect.transform(df_ins_merge["p_insecticidal_raw"].values)), 0.90)

    # OOD Estimator
    df_train = pd.read_parquet(REPEL_FEAT_FILE)
    df_train_labeled = df_train[df_train["repellent_active"].notna()].copy()
    meta_cols = ["compound_id", "canonical_smiles", "source_dataset", "repellent_active", "scaffold_smiles", "fold_id", "qc_status"]
    X_train = df_train_labeled[[c for c in df_train_labeled.columns if c not in meta_cols]].values
    ood_imp = SimpleImputer(strategy="median").fit(X_train)
    X_train_imp = ood_imp.transform(X_train)
    ood_sc = StandardScaler().fit(X_train_imp)
    X_train_std = ood_sc.transform(X_train_imp)

    ood_nn = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=-1).fit(X_train_std)
    train_dist, _ = ood_nn.kneighbors(X_train_std)
    train_mean_dist = train_dist.mean()

    return {
        "m_rep": models_repel, "iso_r": iso_repel, "q_r": q_hat_r,
        "m_ins": models_insect, "iso_i": iso_insect, "q_i": q_hat_i,
        "imp": ood_imp, "sc": ood_sc, "nn": ood_nn, "tm_dist": train_mean_dist
    }

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

def map_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr_m = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr_m)
    arr_r = []
    for _, func in RDKIT_2D_DESCRIPTORS:
        try:
            v = func(mol)
            arr_r.append(np.nan if np.isinf(v) else v)
        except: arr_r.append(np.nan)
    return np.concatenate([arr_m, arr_r]).reshape(1, -1)

# Initialize silently
with st.spinner(""):
    env = load_ecosystem()

# ----------------- UI Dashboard -----------------
st.markdown('<p class="main-title">ECOAI Predictor</p>', unsafe_allow_html=True)

user_smiles = st.text_input("SMILES Sequence", "CC(C)N1CC[C@@]2(C[C@H]3CC[C@@H](C2)N3)OC1=O")

if st.button("Execute", type="primary"):
    feat = map_features(user_smiles)
    if feat is None:
        st.error("Invalid SMILES format. Please verify chemical structure.")
    else:
        with st.spinner("Processing..."):
            f_imp = env["imp"].transform(feat)
            f_std = env["sc"].transform(f_imp)
            dist, _ = env["nn"].kneighbors(f_std)
            ood = dist.mean() / env["tm_dist"]

            p_rep_raw = [m["model"].predict_proba(m["scaler"].transform(m["imputer"].transform(feat)))[0,1] for m in env["m_rep"]]
            p_rep = float(env["iso_r"].transform([np.mean(p_rep_raw)])[0])
            r_l = max(0.0, p_rep - env["q_r"])
            r_u = min(1.0, p_rep + env["q_r"])

            p_ins_raw = [m["model"].predict_proba(m["scaler"].transform(m["imputer"].transform(feat)))[0,1] for m in env["m_ins"]]
            p_ins = float(env["iso_i"].transform([np.mean(p_ins_raw)])[0])
            i_l = max(0.0, p_ins - env["q_i"])
            i_u = min(1.0, p_ins + env["q_i"])

        st.markdown("<hr style='margin: 30px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

        # ----------------- Categories -----------------
        if p_rep > 0.5 and p_ins < 0.2:
            st.markdown('<div class="alert-holy-grail"><h3 style="margin-top:0;">HOLY GRAIL</h3><b>A biologically safe and highly protective spatial repellent barrier.</b></div>', unsafe_allow_html=True)
        elif p_rep > 0.5 and p_ins >= 0.2:
            st.markdown('<div class="alert-toxic"><h3 style="margin-top:0;">TOXIC REPELLENT</h3><b>High spatial repellency, but carries active insecticidal toxicity.</b></div>', unsafe_allow_html=True)
        elif p_rep <= 0.5 and p_ins >= 0.5:
            st.markdown('<div class="alert-bug-spray"><h3 style="margin-top:0;">BUG SPRAY</h3><b>Insecticidal compound possessing zero effective spatial repellency properties.</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-inactive"><h3 style="margin-top:0;">INACTIVE</h3><b>General chemical entity failing to bind effectively to monitored targets.</b></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        p_non_rep = 1.0 - p_rep
        p_non_ins = 1.0 - p_ins

        # ----------------- Repellency Bar (Vertical Stack 1) -----------------
        html_rep = (
            '<div style="background-color: #ffffff; border: 1px solid #e2e8f0; padding: 25px; border-radius: 8px; margin-bottom: 25px;">'
            '<p style="margin-top: 0; font-size: 1.1rem; font-weight: 600; color: #2d3748; margin-bottom: 20px;">Repellency Activity</p>'
            '<div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.9rem; color: #4a5568;">'
            '<span style="font-weight:600; color:#2f855a;">Repellent</span>'
            '<span style="font-weight:600; color:#a0aec0;">Non-Repellent</span>'
            '</div>'
            f'<div style="display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 1.4rem; font-weight: 700;">'
            f'<span style="color: #38a169;">{p_rep*100:.1f}%</span>'
            f'<span style="color: #a0aec0;">{p_non_rep*100:.1f}%</span>'
            '</div>'
            '<div style="width: 100%; background-color: #edf2f7; border-radius: 6px; height: 18px; display: flex; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">'
            f'<div style="width: {p_rep*100}%; background-color: #38a169; height: 100%; transition: width 0.5s;"></div>'
            '</div>'
            f'<p style="text-align: center; margin-top: 15px; margin-bottom: 0; color: #718096; font-size: 0.85rem;">90% Conformal Range: [{r_l*100:.1f}% — {r_u*100:.1f}%]</p>'
            '</div>'
        )
        st.markdown(html_rep, unsafe_allow_html=True)

        # ----------------- Insecticidal Bar (Vertical Stack 2) -----------------
        html_ins = (
            '<div style="background-color: #ffffff; border: 1px solid #e2e8f0; padding: 25px; border-radius: 8px; margin-bottom: 25px;">'
            '<p style="margin-top: 0; font-size: 1.1rem; font-weight: 600; color: #2d3748; margin-bottom: 20px;">Insecticidal Toxicity</p>'
            '<div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.9rem; color: #4a5568;">'
            '<span style="font-weight:600; color:#c53030;">Insecticidal</span>'
            '<span style="font-weight:600; color:#a0aec0;">Non-Insecticidal</span>'
            '</div>'
            f'<div style="display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 1.4rem; font-weight: 700;">'
            f'<span style="color: #e53e3e;">{p_ins*100:.1f}%</span>'
            f'<span style="color: #a0aec0;">{p_non_ins*100:.1f}%</span>'
            '</div>'
            '<div style="width: 100%; background-color: #edf2f7; border-radius: 6px; height: 18px; display: flex; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">'
            f'<div style="width: {p_ins*100}%; background-color: #e53e3e; height: 100%; transition: width 0.5s;"></div>'
            '</div>'
            f'<p style="text-align: center; margin-top: 15px; margin-bottom: 0; color: #718096; font-size: 0.85rem;">90% Conformal Range: [{i_l*100:.1f}% — {i_u*100:.1f}%]</p>'
            '</div>'
        )
        st.markdown(html_ins, unsafe_allow_html=True)

        if ood > 2.0:
            html_ood = (
                '<div style="background-color: #fffaf0; border: 1px solid #feebc8; padding: 15px; border-radius: 8px; text-align: center;">'
                f'<span style="color: #c05621; font-weight: 600;">High Structural Variance ({ood:.2f}x Distance)</span>'
                '<p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #7b341e;">This sequence heavily deviates from the primary training domain.</p>'
                '</div>'
            )
            st.markdown(html_ood, unsafe_allow_html=True)
