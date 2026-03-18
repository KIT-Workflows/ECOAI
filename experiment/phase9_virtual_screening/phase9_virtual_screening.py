# ================================================================
# PHASE 9 — VIRTUAL SCREENING
# ================================================================
# Rank the entire LifeChemicals screening library for repellent
# potential using the best model (V1 or V2).
#
# Input:  experiment/phase4_model_training/models/
#         Datasets/data/features_combined.parquet
#         experiment/phase4_model_training/training_results.json
# Output: Datasets/data/screening_results.parquet
#         experiment/phase9_virtual_screening/screening_report.json
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
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
PHASE9_DIR   = PROJECT_ROOT / "experiment" / "phase9_virtual_screening"
MODEL_DIR    = PROJECT_ROOT / "experiment" / "phase4_model_training" / "models"

# ── Input ──────────────────────────────────────────────────
INPUT_FEATURES     = DATA_DIR / "features_combined.parquet"
INPUT_PREDICTIONS  = DATA_DIR / "model_predictions_v1.parquet"
TRAINING_RESULTS   = PROJECT_ROOT / "experiment" / "phase4_model_training" / "training_results.json"

# ── Output ─────────────────────────────────────────────────
OUTPUT_SCREENING   = DATA_DIR / "screening_results.parquet"
SCREENING_REPORT   = PHASE9_DIR / "screening_report.json"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Verify ─────────────────────────────────────────────────
assert INPUT_FEATURES.exists(), f"❌ {INPUT_FEATURES} — Run Phase 3 first!"
PHASE9_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 9 — VIRTUAL SCREENING")
print("=" * 60)


# ================================================================
# Cell 2 — Load Data and Models
# ================================================================

df = pd.read_parquet(INPUT_FEATURES)

meta_cols = [
    "compound_id", "canonical_smiles", "source_dataset",
    "repellent_active", "scaffold_smiles", "fold_id", "qc_status",
]
feature_cols = [c for c in df.columns if c not in meta_cols]

# Separate labeled and unlabeled
df_labeled   = df[df["repellent_active"].notna()].copy()
df_unlabeled = df[df["repellent_active"].isna()].copy()

print(f"\n✅ Data loaded.")
print(f"   Labeled molecules:   {len(df_labeled)}")
print(f"   Unlabeled (screen):  {len(df_unlabeled)}")
print(f"   Features:            {len(feature_cols)}")

# Determine best classical model
with open(TRAINING_RESULTS, "r") as f:
    training_results = json.load(f)
best_classical_model = training_results.get("best_classical_model", "LightGBM")
print(f"   Best model (Phase 4): {best_classical_model}")

# Load all fold models (Note: Phase 4 saves all classical models as 'lgb_fold_*.pkl' for backwards compatibility)
fold_models = []
for fold in range(5):
    model_path = MODEL_DIR / f"lgb_fold_{fold}.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            fold_models.append(pickle.load(f))
    else:
        print(f"  ⚠️  Model not found: {model_path}")

print(f"  Loaded {len(fold_models)} fold models.")


# ================================================================
# Cell 3 — Ensemble Prediction on Screening Library
# ================================================================
# Average predictions across all fold models for robust estimates.

print("\n" + "=" * 60)
print("  ENSEMBLE SCREENING")
print("=" * 60)

X_screen = df_unlabeled[feature_cols].values
all_fold_preds = []

for fold_idx, model_info in enumerate(fold_models):
    model   = model_info["model"]
    imputer = model_info["imputer"]
    scaler  = model_info["scaler"]

    X_imp = imputer.transform(X_screen)
    X_std = scaler.transform(X_imp)

    preds = model.predict_proba(X_std)[:, 1]
    all_fold_preds.append(preds)
    print(f"  Fold {fold_idx}: predicted {len(preds)} molecules")

# Ensemble mean and std
all_fold_preds = np.array(all_fold_preds)  # (n_folds, n_molecules)
ensemble_mean = all_fold_preds.mean(axis=0)
ensemble_std  = all_fold_preds.std(axis=0)

print(f"\n  Ensemble predictions computed.")
print(f"  P(repellent) range: [{ensemble_mean.min():.4f}, {ensemble_mean.max():.4f}]")
print(f"  Prediction uncertainty (std): mean={ensemble_std.mean():.4f}, "
      f"max={ensemble_std.max():.4f}")


# ================================================================
# Cell 4 — Calibrate Screening Predictions
# ================================================================
# Fit isotonic calibration on the labeled OOF predictions,
# then apply to the screening library.

print("\n── Calibrating screening predictions ──")

# Load labeled OOF predictions
if INPUT_PREDICTIONS.exists():
    df_oof = pd.read_parquet(INPUT_PREDICTIONS)
    y_true_labeled = df_oof["repellent_active"].values.astype(int)
    y_raw_labeled  = df_oof["p_repellent_winner_raw"].values

    # Fit global isotonic calibration on all OOF predictions
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_reg.fit(y_raw_labeled, y_true_labeled)

    calibrated_screen = iso_reg.transform(ensemble_mean)
    calibrated_screen = np.clip(calibrated_screen, 1e-8, 1 - 1e-8)

    print(f"  ✅ Calibrated predictions: [{calibrated_screen.min():.4f}, {calibrated_screen.max():.4f}]")
else:
    print(f"  ⚠️  No OOF predictions found. Using raw ensemble mean.")
    calibrated_screen = ensemble_mean


# ================================================================
# Cell 5 — OOD Scoring for Screening Library
# ================================================================
# Flag molecules that are far from the training distribution.

print("\n── Computing OOD scores ──")

# Fit k-NN on labeled training data
X_labeled = df_labeled[feature_cols].values
imp = SimpleImputer(strategy="median")
X_labeled_imp = imp.fit_transform(X_labeled)
sc = StandardScaler()
X_labeled_std = sc.fit_transform(X_labeled_imp)

K_OOD = 5
nn = NearestNeighbors(n_neighbors=K_OOD, metric="euclidean", n_jobs=-1)
nn.fit(X_labeled_std)

# OOD for screening molecules
X_screen_imp = imp.transform(X_screen)
X_screen_std = sc.transform(X_screen_imp)
distances, _ = nn.kneighbors(X_screen_std)
ood_scores = distances.mean(axis=1)

# Normalize
ood_min, ood_max = ood_scores.min(), ood_scores.max()
if ood_max > ood_min:
    ood_scores_norm = (ood_scores - ood_min) / (ood_max - ood_min)
else:
    ood_scores_norm = np.zeros_like(ood_scores)

print(f"  OOD scores: mean={ood_scores_norm.mean():.4f}, max={ood_scores_norm.max():.4f}")


# ================================================================
# Cell 6 — Conformal Prediction Intervals for Screening
# ================================================================
# Apply conformal prediction using the calibration residuals
# from labeled data as the nonconformity score distribution.

ALPHA = 0.10  # 90% coverage

if INPUT_PREDICTIONS.exists():
    # Calibrated residuals from labeled OOF
    calibrated_labeled = iso_reg.transform(y_raw_labeled)
    residuals = np.abs(y_true_labeled - calibrated_labeled)

    n_cal = len(residuals)
    q_level = np.ceil((1 - ALPHA) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(residuals, q_level)

    conformal_lower = np.clip(calibrated_screen - q_hat, 0, 1)
    conformal_upper = np.clip(calibrated_screen + q_hat, 0, 1)
else:
    conformal_lower = np.clip(calibrated_screen - ensemble_std * 2, 0, 1)
    conformal_upper = np.clip(calibrated_screen + ensemble_std * 2, 0, 1)

interval_widths = conformal_upper - conformal_lower
print(f"\n  Conformal intervals: mean width={interval_widths.mean():.4f}")


# ================================================================
# Cell 7 — Assemble Screening Results
# ================================================================

df_screen = df_unlabeled[["compound_id", "canonical_smiles",
                           "source_dataset", "scaffold_smiles"]].copy()

df_screen["p_repellent"]      = calibrated_screen
df_screen["p_non_repellent"]  = 1.0 - calibrated_screen
df_screen["ensemble_std"]     = ensemble_std
df_screen["prediction_interval_lower"] = conformal_lower
df_screen["prediction_interval_upper"] = conformal_upper
df_screen["prediction_interval"] = [
    f"[{lo:.4f}, {up:.4f}]"
    for lo, up in zip(conformal_lower, conformal_upper)
]
df_screen["ood_score"] = ood_scores_norm

# Rank by p_repellent (descending)
df_screen = df_screen.sort_values("p_repellent", ascending=False).reset_index(drop=True)
df_screen["rank"] = range(1, len(df_screen) + 1)

# Flag high-uncertainty predictions
ood_threshold = np.percentile(ood_scores_norm, 90)
df_screen["high_ood"] = ood_scores_norm > ood_threshold
df_screen["high_uncertainty"] = ensemble_std > np.percentile(ensemble_std, 90)

print(f"\n✅ Screening results assembled: {len(df_screen)} molecules")
print(f"   High OOD flagged:         {df_screen['high_ood'].sum()}")
print(f"   High uncertainty flagged: {df_screen['high_uncertainty'].sum()}")


# ================================================================
# Cell 8 — Top Candidates
# ================================================================

print("\n" + "=" * 60)
print("  TOP 20 REPELLENT CANDIDATES")
print("=" * 60)

top20 = df_screen.head(20)
print(top20[["rank", "compound_id", "p_repellent", "ensemble_std",
             "ood_score", "prediction_interval"]].to_string(index=False))

# Also show candidates with highest uncertainty
print(f"\n  ── Top 10 Most Uncertain Predictions ──")
uncertain = df_screen.nlargest(10, "ensemble_std")
print(uncertain[["rank", "compound_id", "p_repellent", "ensemble_std",
                  "ood_score"]].to_string(index=False))


# ================================================================
# Cell 9 — Screening Distribution Visualization
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) p_repellent distribution
axes[0].hist(calibrated_screen, bins=50, color="#4A90D9", alpha=0.8, edgecolor="white")
axes[0].set_xlabel("p(repellent)")
axes[0].set_ylabel("Count")
axes[0].set_title("Predicted Repellent Probability")
axes[0].axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="threshold=0.5")
axes[0].legend()

# (2) OOD score distribution
axes[1].hist(ood_scores_norm, bins=50, color="#E67E22", alpha=0.8, edgecolor="white")
axes[1].set_xlabel("OOD Score")
axes[1].set_ylabel("Count")
axes[1].set_title("Out-of-Distribution Score")
axes[1].axvline(x=ood_threshold, color="red", linestyle="--", alpha=0.5,
                label=f"90th pctl={ood_threshold:.2f}")
axes[1].legend()

# (3) p_repellent vs. uncertainty
axes[2].scatter(calibrated_screen, ensemble_std, alpha=0.3, s=5, c="#27AE60")
axes[2].set_xlabel("p(repellent)")
axes[2].set_ylabel("Ensemble Std (uncertainty)")
axes[2].set_title("Prediction vs. Uncertainty")

fig.suptitle("Virtual Screening — LifeChemicals Library",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PHASE9_DIR / "screening_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📊 Saved: screening_distributions.png")


# ================================================================
# Cell 10 — Save Screening Results
# ================================================================

df_screen.to_parquet(OUTPUT_SCREENING, index=False, engine="pyarrow")
print(f"\n✅ Saved: {OUTPUT_SCREENING}")
print(f"   Size: {OUTPUT_SCREENING.stat().st_size / 1024:.1f} KB")

# CSV for easy sharing
csv_top100 = PHASE9_DIR / "top_100_candidates.csv"
df_screen.head(100).to_csv(csv_top100, index=False)
print(f"✅ Saved: {csv_top100}")


# ================================================================
# Cell 11 — Screening Report
# ================================================================

report = {
    "pipeline": "Phase 9 — Virtual Screening",
    "library_screened": "LifeChemicals insecticide library",
    "n_molecules_screened": len(df_screen),
    "model_used": f"{best_classical_model} ensemble (5-fold)",
    "calibration": "isotonic regression",
    "prediction_summary": {
        "p_repellent_mean":   float(calibrated_screen.mean()),
        "p_repellent_median": float(np.median(calibrated_screen)),
        "p_repellent_std":    float(calibrated_screen.std()),
        "predicted_actives_above_0.5": int((calibrated_screen >= 0.5).sum()),
        "predicted_actives_above_0.7": int((calibrated_screen >= 0.7).sum()),
        "predicted_actives_above_0.9": int((calibrated_screen >= 0.9).sum()),
    },
    "uncertainty": {
        "mean_ensemble_std":    float(ensemble_std.mean()),
        "mean_interval_width":  float(interval_widths.mean()),
        "high_ood_count":       int(df_screen["high_ood"].sum()),
        "high_uncertainty_count": int(df_screen["high_uncertainty"].sum()),
    },
    "output_files": {
        "full_results": str(OUTPUT_SCREENING),
        "top_100_csv":  str(csv_top100),
    },
}

with open(SCREENING_REPORT, "w") as f:
    json.dump(report, f, indent=2)

print(f"✅ Report saved: {SCREENING_REPORT}")


# ================================================================
# Cell 12 — Final Summary
# ================================================================

print("\n" + "=" * 60)
print("  PHASE 9 COMPLETE — Virtual Screening")
print("=" * 60)
print(f"\n  Screened: {len(df_screen)} LifeChemicals molecules")
print(f"  Predicted actives (p ≥ 0.5): {(calibrated_screen >= 0.5).sum()}")
print(f"  Predicted actives (p ≥ 0.7): {(calibrated_screen >= 0.7).sum()}")
print(f"  High-confidence candidates:   see top_100_candidates.csv")
print(f"\n  Artifacts:")
print(f"    1. {OUTPUT_SCREENING}")
print(f"    2. {csv_top100}")
print(f"    3. screening_distributions.png")
print(f"    4. {SCREENING_REPORT}")
print(f"\n  Ready for Phase 10 (V3 Multitask) →")
print("=" * 60)
