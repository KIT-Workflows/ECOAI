# ================================================================
# PHASE 6 — INTERPRETABILITY
# ================================================================
# Post-hoc interpretability using SHAP and feature importance
# analysis. Verify that the model's decision drivers are
# chemically plausible (not dataset/source artifacts).
#
# Input:  experiment/phase4_model_training/models/
#         Datasets/data/features_combined.parquet
#         experiment/phase4_model_training/training_results.json
# Output: experiment/phase6_interpretability/ (plots + report)
#
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# !pip install shap lightgbm scikit-learn pandas pyarrow numpy matplotlib


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
PHASE6_DIR   = PROJECT_ROOT / "experiment" / "phase6_interpretability"
MODEL_DIR    = PROJECT_ROOT / "experiment" / "phase4_model_training" / "models"
RESULTS_FILE = PROJECT_ROOT / "experiment" / "phase4_model_training" / "training_results.json"

# ── Input ──────────────────────────────────────────────────
INPUT_FEATURES = DATA_DIR / "features_combined.parquet"

# ── Verify ─────────────────────────────────────────────────
assert INPUT_FEATURES.exists(), f"❌ {INPUT_FEATURES} — Run Phase 3 first!"
PHASE6_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 6 — INTERPRETABILITY (SHAP)")
print("=" * 60)


# ================================================================
# Cell 2 — Load Data and Models
# ================================================================

df = pd.read_parquet(INPUT_FEATURES)
with open(RESULTS_FILE, "r") as f:
    training_results = json.load(f)

selected_model = training_results["selected_model"]
best_classical = training_results.get("best_classical_model", "LightGBM")
print(f"\n✅ Selected model: {selected_model}")
print(f"   Best classical model (analyzed here): {best_classical}")

# ── Identify feature columns ──────────────────────────────
meta_cols = [
    "compound_id", "canonical_smiles", "source_dataset",
    "repellent_active", "scaffold_smiles", "fold_id", "qc_status",
]
feature_cols = [c for c in df.columns if c not in meta_cols]

# ── Filter to labeled + QC-pass ───────────────────────────
df_labeled = df[
    (df["qc_status"] == "pass") &
    (df["repellent_active"].notna())
].copy()

X = df_labeled[feature_cols].values
y = df_labeled["repellent_active"].values.astype(int)

print(f"   Molecules: {len(df_labeled)}")
print(f"   Features:  {len(feature_cols)}")


# ================================================================
# Cell 3 — Native Feature Importance
# ================================================================
# Even if FT-Transformer was selected overall, we analyze the best
# classical model's feature importance as an interpretability signal.

print("\n" + "=" * 60)
print(f"  {best_classical.upper()} FEATURE IMPORTANCE")
print("=" * 60)

# Load fold 0 model as representative
with open(MODEL_DIR / "lgb_fold_0.pkl", "rb") as f:
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
print(f"\n  Top {top_n} features by gain importance:")
for i, (_, row) in enumerate(top_features.iterrows()):
    print(f"    {i+1:>2d}. {row['feature']:35s}  {row['importance']:>8.0f}")

# Separate Morgan FP bits vs. RDKit 2D descriptors in top features
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
ax.set_xlabel("Feature Importance (gain)")
ax.set_title(f"Top {top_n} Features — {best_classical} (Fold 0)")
fig.tight_layout()
fig.savefig(PHASE6_DIR / "native_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  📊 Saved: native_feature_importance.png")


# ================================================================
# Cell 4 — SHAP Analysis
# ================================================================
# TreeExplainer is very efficient for tree-based models and gives
# exact SHAP values.

print("\n" + "=" * 60)
print("  SHAP ANALYSIS")
print("=" * 60)

# Preprocess features the same way as training
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imp = model_info["imputer"]
sc  = model_info["scaler"]

X_imp = imp.transform(X)
X_std = sc.transform(X_imp)

# Create a DataFrame with feature names for SHAP plots
X_df = pd.DataFrame(X_std, columns=feature_cols)

# ── XGBoost + SHAP Compatibility Fix ──
# XGBoost >= 2.1 saves base_score as a string representation of a list (e.g., "[0.5]") 
# inside its binary UBJSON format, which breaks SHAP's TreeExplainer parser natively.
# We monkey-patch the UBJSON decoder inside SHAP to intercept and fix this on the fly.
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

# For binary classification, shap_values may be a list [class_0, class_1]
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # class 1 = repellent
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
plt.savefig(PHASE6_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  📊 Saved: shap_summary_beeswarm.png")


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
plt.savefig(PHASE6_DIR / "shap_mean_abs_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  📊 Saved: shap_mean_abs_bar.png")


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

# Save to CSV
shap_importance.to_csv(PHASE6_DIR / "shap_feature_importance.csv", index=False)
print(f"\n  📊 Saved: shap_feature_importance.csv")


# ================================================================
# Cell 8 — Chemical Plausibility Check
# ================================================================
# Verify that the most important features are chemically sensible.
# Flag any features that might be artifacts (e.g., source-related).

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

print(f"\n  ℹ️  Morgan FP bits correspond to specific circular substructures.")
print(f"     To decode which substructure a bit represents, use:")
print(f"     AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=info)")


# ================================================================
# Cell 9 — SHAP Dependence Plots for Top Features
# ================================================================
# Show how the top features influence predictions.

print("\n── SHAP Dependence Plots ──")

# Top 6 non-Morgan features for dependence plots
top_dep_features = shap_importance[
    ~shap_importance["feature"].str.startswith("mfp_")
].head(6)["feature"].tolist()

if len(top_dep_features) > 0:
    n_plots = min(6, len(top_dep_features))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top_dep_features[:n_plots]):
        feat_idx = feature_cols.index(feat)
        shap.dependence_plot(
            feat_idx,
            shap_vals,
            X_df,
            ax=axes[i],
            show=False,
        )
        axes[i].set_title(feat, fontsize=10)

    # Hide unused axes
    for i in range(n_plots, 6):
        axes[i].set_visible(False)

    fig.suptitle("SHAP Dependence Plots — Top 2D Descriptors",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PHASE6_DIR / "shap_dependence_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: shap_dependence_plots.png")


# ================================================================
# Cell 10 — Save Interpretability Report
# ================================================================

report = {
    "pipeline": "Phase 6 — Interpretability",
    "model_analyzed": f"{best_classical} (fold 0)",
    "selected_model": selected_model,
    "shap_method": "TreeExplainer (exact)",
    "top_20_features_by_shap": shap_importance.head(20).to_dict("records"),
    "top_20_2d_descriptors": top_2d.to_dict("records"),
    "morgan_fp_in_top_30": int(
        shap_importance.head(30)["feature"].str.startswith("mfp_").sum()
    ),
    "rdkit_2d_in_top_30": int(
        (~shap_importance.head(30)["feature"].str.startswith("mfp_")).sum()
    ),
    "plots_generated": [
        "native_feature_importance.png",
        "shap_summary_beeswarm.png",
        "shap_mean_abs_bar.png",
        "shap_dependence_plots.png",
    ],
}

with open(PHASE6_DIR / "interpretability_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n✅ Report saved: interpretability_report.json")


# ================================================================
# Cell 11 — Final Summary
# ================================================================

print("\n" + "=" * 60)
print("  PHASE 6 COMPLETE — Interpretability")
print("=" * 60)
print(f"\n  Key findings:")
print(f"    • Morgan FP bits in top 30: "
      f"{report['morgan_fp_in_top_30']}")
print(f"    • RDKit 2D descriptors in top 30: "
      f"{report['rdkit_2d_in_top_30']}")
print(f"\n  Artifacts:")
print(f"    1. shap_summary_beeswarm.png")
print(f"    2. shap_mean_abs_bar.png")
print(f"    3. shap_dependence_plots.png")
print(f"    4. native_feature_importance.png")
print(f"    5. shap_feature_importance.csv")
print(f"    6. interpretability_report.json")
print(f"\n  Review the plots to confirm features are chemically")
print(f"  plausible and not dataset/source artifacts.")
print(f"\n  Ready for Phase 7 (Quantum Descriptors) →")
print("=" * 60)
