# ================================================================
# STEP 8 — INTERPRETABILITY (Insecticidal Pipeline)
# ================================================================
# Post-hoc interpretability using SHAP and feature importance
# analysis on the best classical model (e.g., RandomForest).
# Verify the chemical drivers of insecticidal activity.
#
# Input:  Datasets/data/phase10_artifacts/insecticide_features.npz
#         Datasets/data/phase10_artifacts/insecticide_meta.parquet
#         Datasets/data/phase10_artifacts/winner_results.json
#         implementation/models/classical_fold_*.pkl
# Output: implementation/artifacts/shap/ (plots + report)
#
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
IMPL_DIR     = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation"
ARTIFACT_DIR = DATA_DIR / "phase10_artifacts"
MODEL_DIR    = IMPL_DIR / "models"
SHAP_DIR     = ARTIFACT_DIR / "shap"

# ── Input ──────────────────────────────────────────────────
INPUT_FEATURES = ARTIFACT_DIR / "insecticide_features.npz"
INPUT_META     = ARTIFACT_DIR / "insecticide_meta.parquet"
INPUT_WINNER   = ARTIFACT_DIR / "winner_results.json"
FEATURE_COLS   = ARTIFACT_DIR / "feature_columns.json"

# ── Verify ─────────────────────────────────────────────────
assert INPUT_FEATURES.exists(), f"❌ {INPUT_FEATURES} — Run earlier steps!"
assert INPUT_WINNER.exists(), f"❌ {INPUT_WINNER} — Run step 6 first!"

SHAP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  STEP 8 — INTERPRETABILITY (SHAP)")
print("=" * 60)


# ================================================================
# Cell 2 — Load Data and Models
# ================================================================

npz = np.load(INPUT_FEATURES)
X_all = npz["X"]
df_meta = pd.read_parquet(INPUT_META)

with open(INPUT_WINNER, "r") as f:
    winner_res = json.load(f)

with open(FEATURE_COLS, "r") as f:
    feat_meta = json.load(f)

feature_cols = feat_meta["all_cols"]

best_classical = winner_res["best_classical_name"]
print(f"\n✅ Best classical model (analyzed here): {best_classical}")

y = df_meta["insecticidal_active"].values.astype(int)

print(f"   Molecules: {len(df_meta)}")
print(f"   Features:  {len(feature_cols)}")


# ================================================================
# Cell 3 — Native Feature Importance
# ================================================================
# We analyze the classical model's fold 0 feature importance

print("\n" + "=" * 60)
print(f"  {best_classical.upper()} FEATURE IMPORTANCE")
print("=" * 60)

# Load fold 0 model
with open(MODEL_DIR / "classical_fold_0.pkl", "rb") as f:
    model_info = pickle.load(f)

classical_model = model_info["model"]

# Gain-based importance (or Gini for RF)
importances = classical_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances,
}).sort_values("importance", ascending=False)

# Top 30 features
top_n = 30
top_features = importance_df.head(top_n)
print(f"\n  Top {top_n} features by gain/Gini importance:")
for i, (_, row) in enumerate(top_features.iterrows()):
    print(f"    {i+1:>2d}. {row['feature']:35s}  {row['importance']:>8.0f}")

# Separate Morgan FP bits vs. RDKit 2D descriptors
n_morgan_in_top = sum(1 for f in top_features["feature"] if f.startswith("mfp_"))
n_rdkit_in_top  = top_n - n_morgan_in_top
print(f"\n  Top {top_n} breakdown: {n_morgan_in_top} Morgan FP bits, {n_rdkit_in_top} RDKit 2D descriptors")

# Save importance plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    range(top_n),
    top_features["importance"].values[::-1],
    color="#4A90D9",
    alpha=0.8,
)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features["feature"].values[::-1], fontsize=8)
ax.set_xlabel("Feature Importance (gain/Gini)")
ax.set_title(f"Top {top_n} Features — {best_classical} (Fold 0)")
fig.tight_layout()
fig.savefig(SHAP_DIR / "native_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  📊 Saved: shap/native_feature_importance.png")


# ================================================================
# Cell 4 — SHAP Analysis
# ================================================================

print("\n" + "=" * 60)
print("  SHAP ANALYSIS")
print("=" * 60)

# Preprocess features the same way as training
imp = model_info["imputer"]
sc  = model_info["scaler"]

X_imp = imp.transform(X_all)
X_std = sc.transform(X_imp)

X_df = pd.DataFrame(X_std, columns=feature_cols)

# ── XGBoost + SHAP Compatibility Fix ──
if type(classical_model).__name__ == "XGBClassifier":
    import shap.explainers._tree
    
    if hasattr(shap.explainers._tree, "decode_ubjson_buffer"):
        orig_decode = shap.explainers._tree.decode_ubjson_buffer
        
        def patched_decode(fd):
            res = orig_decode(fd)
            if isinstance(res, dict) and "learner" in res:
                bs = res.get("learner", {}).get("learner_model_param", {}).get("base_score", "")
                if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                    res["learner"]["learner_model_param"]["base_score"] = bs[1:-1]
            return res
            
        shap.explainers._tree.decode_ubjson_buffer = patched_decode
        print("  🔧 Applied XGBoost + SHAP UBJSON compatibility fix.")

# SHAP TreeExplainer
print(f"  Computing SHAP values for {best_classical} (this may take a moment)...")
explainer = shap.TreeExplainer(classical_model)
shap_values = explainer.shap_values(X_std)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # class 1 = active
elif len(shap_values.shape) == 3:
    shap_vals = shap_values[:, :, 1] # SHAP returns (samples, features, classes)
else:
    shap_vals = shap_values

print(f"  ✅ SHAP values computed: shape = {shap_vals.shape}")


# ================================================================
# Cell 5 — SHAP Summary Plot (Beeswarm)
# ================================================================

print("\n── SHAP Summary Plot ──")

fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(
    shap_vals,
    X_df,
    max_display=25,
    show=False,
    plot_size=None,
)
plt.title("SHAP Summary — Top 25 Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  📊 Saved: shap/shap_summary_beeswarm.png")


# ================================================================
# Cell 6 — SHAP Bar Plot (Mean Absolute SHAP)
# ================================================================

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_vals,
    X_df,
    plot_type="bar",
    max_display=25,
    show=False,
    plot_size=None,
)
plt.title("Mean |SHAP| — Top 25 Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_mean_abs_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  📊 Saved: shap/shap_mean_abs_bar.png")


# ================================================================
# Cell 7 — SHAP Feature Importance Table
# ================================================================

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
shap_importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False)

print("\n  Top 30 features by mean |SHAP|:")
for i, (_, row) in enumerate(shap_importance.head(30).iterrows()):
    feat_type = "MFP" if row["feature"].startswith("mfp_") else "2D"
    print(f"    {i+1:>2d}. [{feat_type:3s}] {row['feature']:35s}  {row['mean_abs_shap']:.6f}")

shap_importance.to_csv(SHAP_DIR / "shap_feature_importance.csv", index=False)
print(f"\n  📊 Saved: shap/shap_feature_importance.csv")


# ================================================================
# Cell 8 — Chemical Plausibility Check
# ================================================================

print("\n" + "=" * 60)
print("  CHEMICAL PLAUSIBILITY CHECK")
print("=" * 60)

# Known chemically meaningful RDKit descriptors
KNOWN_MEANINGFUL = {
    "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "MolWt", "ExactMolWt", "NumAromaticRings",
    "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms",
    "LabuteASA", "BalabanJ", "BertzCT", "MolMR",
    "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
    "NHOHCount", "NOCount", "RingCount",
}

# Check top 20 non-MFP features
top_2d = shap_importance[~shap_importance["feature"].str.startswith("mfp_")].head(20)
print("\n  Top 20 RDKit 2D descriptors by SHAP importance:")
for i, (_, row) in enumerate(top_2d.iterrows()):
    feat = row["feature"]
    is_known = "✅" if feat in KNOWN_MEANINGFUL else "⚠️"
    print(f"    {i+1:>2d}. {is_known} {feat:35s}  SHAP={row['mean_abs_shap']:.6f}")

# Morgan FP bits in top 20
top_mfp = shap_importance[shap_importance["feature"].str.startswith("mfp_")].head(20)
print(f"\n  Top 20 Morgan FP bits (substructure patterns):")
for i, (_, row) in enumerate(top_mfp.iterrows()):
    bit_id = row["feature"].replace("mfp_", "")
    print(f"    {i+1:>2d}. Bit {bit_id:>5s}  SHAP={row['mean_abs_shap']:.6f}")

print(f"\n  ➡️ To understand a bit, use: view_morgan_bit(mol, {bit_id}) in RDKit.")
print("\n  → End of Interpretability.")
print("=" * 60)
