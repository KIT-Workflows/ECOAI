# ================================================================
# PHASE 8 — V2 ABLATION STUDY
# ================================================================
# Compare three feature sets on the labeled quantum subset:
#   1. Cheminformatics-only (Morgan FP + RDKit 2D)
#   2. Quantum-only (xTB / DFT descriptors)
#   3. Fused (cheminformatics + quantum)
#
# The fused model is promoted to V2 ONLY if it improves both
# discrimination (PR-AUC) AND calibration (Brier score).
#
# Input:  Datasets/data/quantum_descriptors.parquet
#         Datasets/data/features_combined.parquet
#         Datasets/data/split_manifest.json
# Output: experiment/phase8_v2_ablation/ablation_results.json
#         Datasets/data/model_predictions_v2.parquet (if fused wins)
#
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# !pip install lightgbm scikit-learn pandas pyarrow numpy matplotlib


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
PHASE8_DIR   = PROJECT_ROOT / "experiment" / "phase8_v2_ablation"

# ── Input ──────────────────────────────────────────────────
INPUT_QUANTUM  = DATA_DIR / "quantum_descriptors.parquet"
INPUT_FEATURES = DATA_DIR / "features_combined.parquet"
INPUT_MANIFEST = DATA_DIR / "split_manifest.json"

# ── Output ─────────────────────────────────────────────────
ABLATION_RESULTS = PHASE8_DIR / "ablation_results.json"
OUTPUT_V2_PREDS  = DATA_DIR / "model_predictions_v2.parquet"
RESULTS_FILE     = PROJECT_ROOT / "experiment" / "phase4_model_training" / "training_results.json"

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS = 5
np.random.seed(RANDOM_SEED)

# ── Verify ─────────────────────────────────────────────────
assert INPUT_QUANTUM.exists(),  f"❌ {INPUT_QUANTUM} — Run Phase 7 first!"
assert INPUT_FEATURES.exists(), f"❌ {INPUT_FEATURES} — Run Phase 3 first!"
PHASE8_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 8 — V2 ABLATION STUDY")
print("=" * 60)


# ================================================================
# Cell 2 — Load and Merge Data
# ================================================================

df_quantum = pd.read_parquet(INPUT_QUANTUM)
df_features = pd.read_parquet(INPUT_FEATURES)

# Determine the best classical model from Phase 4
with open(RESULTS_FILE, "r") as f:
    training_results = json.load(f)
best_classical_model = training_results.get("best_classical_model", "LightGBM")

print(f"\n✅ Base model for ablation: {best_classical_model}")

# Identify quantum feature columns
quantum_feature_cols = [
    "homo_ev", "lumo_ev", "homo_lumo_gap_ev", "total_energy_eh",
    "dipole_debye", "partial_charge_mean", "partial_charge_std",
    "partial_charge_max", "partial_charge_min", "mmff_energy",
]
# Only keep columns that exist and have non-null values
quantum_feature_cols = [
    c for c in quantum_feature_cols
    if c in df_quantum.columns and df_quantum[c].notna().any()
]

print(f"\n  Quantum descriptors loaded: {len(df_quantum)} molecules")
print(f"  Available quantum features: {quantum_feature_cols}")

# ── Filter to labeled molecules with converged QC ─────────
df_q_labeled = df_quantum[
    df_quantum["repellent_active"].notna() &
    df_quantum["convergence"].isin(["converged", "na_fallback"])
].copy()

print(f"  Labeled with converged QC: {len(df_q_labeled)}")

# ── Cheminformatics features for this same subset ─────────
meta_cols = [
    "compound_id", "canonical_smiles", "source_dataset",
    "repellent_active", "scaffold_smiles", "fold_id", "qc_status",
]
chem_feature_cols = [c for c in df_features.columns if c not in meta_cols]

df_chem_subset = df_features[
    df_features["compound_id"].isin(df_q_labeled["compound_id"])
].copy()

# Merge cheminformatics and quantum features
df_merged = df_chem_subset.merge(
    df_q_labeled[["compound_id"] + quantum_feature_cols],
    on="compound_id",
    how="inner",
)

y = df_merged["repellent_active"].values.astype(int)
print(f"\n  Merged dataset: {len(df_merged)} molecules")
print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"  Cheminformatics features: {len(chem_feature_cols)}")
print(f"  Quantum features: {len(quantum_feature_cols)}")
print(f"  Fused features: {len(chem_feature_cols) + len(quantum_feature_cols)}")


# ================================================================
# Cell 3 — Define Feature Sets
# ================================================================

feature_sets = {
    "cheminformatics_only": chem_feature_cols,
    "quantum_only":         quantum_feature_cols,
    "fused":                chem_feature_cols + quantum_feature_cols,
}

for name, cols in feature_sets.items():
    print(f"  {name:25s}: {len(cols)} features")


# ================================================================
# Cell 4 — Scaffold-Aware CV on Quantum Subset
# ================================================================
# Since the quantum subset may not perfectly map to the original
# fold assignments, we create new scaffold-aware splits for this
# specific subset.

from collections import defaultdict

def scaffold_split_subset(df, n_folds=5, seed=42):
    """Quick scaffold split for the quantum subset."""
    rng = np.random.RandomState(seed)
    fold_ids = np.zeros(len(df), dtype=int)

    scaffold_groups = defaultdict(list)
    for i, (_, row) in enumerate(df.iterrows()):
        sc = row.get("scaffold_smiles", "unknown")
        if pd.isna(sc):
            sc = f"unknown_{i}"
        scaffold_groups[sc].append(i)

    groups = list(scaffold_groups.values())
    groups.sort(key=len, reverse=True)
    rng.shuffle(groups)

    fold_sizes = np.zeros(n_folds, dtype=int)
    for group in groups:
        target = int(np.argmin(fold_sizes))
        for idx in group:
            fold_ids[idx] = target
        fold_sizes[target] += len(group)

    return fold_ids


fold_ids = scaffold_split_subset(df_merged, n_folds=N_FOLDS, seed=RANDOM_SEED)
df_merged["ablation_fold_id"] = fold_ids

print(f"\n  Scaffold-aware {N_FOLDS}-fold split for ablation subset:")
for fold in range(N_FOLDS):
    n = (fold_ids == fold).sum()
    print(f"    Fold {fold}: {n} molecules")


# ================================================================
# Cell 5 — Run Ablation Experiment
# We train the model type identified as best in Phase 4 using
# conservative hyperparameters (since the quantum subset is small).

print("\n" + "=" * 60)
print(f"  ABLATION EXPERIMENT ({best_classical_model})")
print("=" * 60)

ablation_results = {}

for fs_name, fs_cols in feature_sets.items():
    print(f"\n  ── {fs_name} ({len(fs_cols)} features) ──")

    X = df_merged[fs_cols].values
    oof_preds = np.zeros(len(y))
    fold_metrics = []

    for fold in range(N_FOLDS):
        train_mask = fold_ids != fold
        val_mask   = fold_ids == fold

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]

        # Impute + scale
        imp = SimpleImputer(strategy="median")
        X_train_imp = imp.fit_transform(X_train)
        X_val_imp   = imp.transform(X_val)

        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train_imp)
        X_val_std   = sc.transform(X_val_imp)

        # Train
        if best_classical_model == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, verbosity=0, n_jobs=-1,
                eval_metric="logloss", early_stopping_rounds=30
            )
            model.fit(X_train_std, y_train, eval_set=[(X_val_std, y_val)], verbose=False)
        elif best_classical_model == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_split=5, min_samples_leaf=2,
                class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
            )
            model.fit(X_train_std, y_train)
        elif best_classical_model == "LightGBM-DART":
            model = lgb.LGBMClassifier(
                boosting_type="dart", n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
            )
            model.fit(X_train_std, y_train)
        else: # LightGBM fallback
            model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=20,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1,
            )
            model.fit(X_train_std, y_train, eval_set=[(X_val_std, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])

        y_pred_proba = model.predict_proba(X_val_std)[:, 1]
        oof_preds[val_mask] = y_pred_proba

        y_pred_class = (y_pred_proba >= 0.5).astype(int)
        fold_metrics.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y_val, y_pred_proba)),
            "pr_auc":  float(average_precision_score(y_val, y_pred_proba)),
            "mcc":     float(matthews_corrcoef(y_val, y_pred_class)),
            "brier":   float(brier_score_loss(y_val, y_pred_proba)),
        })

    # Overall OOF metrics
    oof_class = (oof_preds >= 0.5).astype(int)
    overall = {
        "roc_auc": float(roc_auc_score(y, oof_preds)),
        "pr_auc":  float(average_precision_score(y, oof_preds)),
        "mcc":     float(matthews_corrcoef(y, oof_class)),
        "brier":   float(brier_score_loss(y, oof_preds)),
    }

    ablation_results[fs_name] = {
        "n_features": len(fs_cols),
        "overall": overall,
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_preds.tolist(),
    }

    print(f"    ROC-AUC: {overall['roc_auc']:.4f}  |  "
          f"PR-AUC: {overall['pr_auc']:.4f}  |  "
          f"MCC: {overall['mcc']:.4f}  |  "
          f"Brier: {overall['brier']:.4f}")


# ================================================================
# Cell 6 — Compare Feature Sets
# ================================================================

print("\n" + "=" * 60)
print("  ABLATION COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    name: result["overall"]
    for name, result in ablation_results.items()
}).T

comparison_df["n_features"] = [
    ablation_results[name]["n_features"]
    for name in comparison_df.index
]

print(comparison_df.to_string(float_format="{:.4f}".format))


# ================================================================
# Cell 7 — Promotion Decision
# ================================================================
# The fused model is promoted ONLY if it improves BOTH:
#   1. Discrimination (PR-AUC) over cheminformatics-only
#   2. Calibration (Brier score) over cheminformatics-only

print("\n" + "=" * 60)
print("  PROMOTION DECISION")
print("=" * 60)

chem_prauc = ablation_results["cheminformatics_only"]["overall"]["pr_auc"]
chem_brier = ablation_results["cheminformatics_only"]["overall"]["brier"]
fused_prauc = ablation_results["fused"]["overall"]["pr_auc"]
fused_brier = ablation_results["fused"]["overall"]["brier"]

prauc_gain = fused_prauc - chem_prauc
brier_gain = chem_brier - fused_brier   # positive = improvement (lower is better)

print(f"  Cheminformatics-only: PR-AUC = {chem_prauc:.4f}, Brier = {chem_brier:.4f}")
print(f"  Fused:                PR-AUC = {fused_prauc:.4f}, Brier = {fused_brier:.4f}")
print(f"\n  PR-AUC gain:  {prauc_gain:+.4f}  {'✅' if prauc_gain > 0 else '❌'}")
print(f"  Brier gain:   {brier_gain:+.4f}  {'✅' if brier_gain > 0 else '❌'}")

promote_fused = (prauc_gain > 0) and (brier_gain > 0)

if promote_fused:
    print(f"\n  🏆 DECISION: PROMOTE fused model to V2!")
    print(f"     Fused features improve BOTH discrimination AND calibration.")
    selected_v2 = "fused"
else:
    print(f"\n  ⏸️  DECISION: STICK WITH V1 (cheminformatics-only).")
    if prauc_gain <= 0:
        print(f"     Fused does NOT improve PR-AUC.")
    if brier_gain <= 0:
        print(f"     Fused does NOT improve Brier score.")
    selected_v2 = "cheminformatics_only"


# ================================================================
# Cell 8 — Comparison Visualization
# ================================================================

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

metrics = ["roc_auc", "pr_auc", "mcc", "brier"]
titles  = ["ROC-AUC ↑", "PR-AUC ↑", "MCC ↑", "Brier Score ↓"]
colors  = ["#4A90D9", "#E67E22", "#27AE60"]

for ax, metric, title in zip(axes, metrics, titles):
    values = [ablation_results[fs]["overall"][metric] for fs in feature_sets]
    bars = ax.bar(list(feature_sets.keys()), values, color=colors, alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("V2 Ablation Study — Feature Set Comparison",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PHASE8_DIR / "ablation_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📊 Saved: ablation_comparison.png")


# ================================================================
# Cell 9 — Save Ablation Results
# ================================================================

ablation_output = {
    "pipeline":       "Phase 8 — V2 Ablation Study",
    "random_seed":    RANDOM_SEED,
    "n_folds":        N_FOLDS,
    "dataset_size":   len(df_merged),
    "feature_sets": {
        name: {"n_features": len(cols), "features": cols}
        for name, cols in feature_sets.items()
    },
    "results":        {k: {kk: vv for kk, vv in v.items() if kk != "oof_predictions"}
                       for k, v in ablation_results.items()},
    "promotion_decision": {
        "promoted_to_v2":     promote_fused,
        "selected_feature_set": selected_v2,
        "prauc_gain":         float(prauc_gain),
        "brier_gain":         float(brier_gain),
    },
}

# Remove oof_predictions from saved results to keep JSON small
with open(ABLATION_RESULTS, "w") as f:
    json.dump(ablation_output, f, indent=2, default=str)

print(f"✅ Ablation results saved: {ABLATION_RESULTS}")


# ================================================================
# Cell 10 — Final Summary
# ================================================================

print("\n" + "=" * 60)
print("  PHASE 8 COMPLETE — V2 Ablation Study")
print("=" * 60)
print(f"\n  Comparison (on {len(df_merged)} labeled quantum molecules):")
print(comparison_df[["n_features", "pr_auc", "brier"]].to_string(
    float_format="{:.4f}".format))
print(f"\n  Decision: {'🏆 FUSED → V2' if promote_fused else '⏸️ Stay with V1'}")
print(f"\n  Artifacts:")
print(f"    1. {ABLATION_RESULTS}")
print(f"    2. ablation_comparison.png")
print(f"\n  Ready for Phase 9 (Virtual Screening) →")
print("=" * 60)
