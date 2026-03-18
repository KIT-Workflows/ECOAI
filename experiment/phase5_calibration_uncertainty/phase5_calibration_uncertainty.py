# ================================================================
# PHASE 5 — CALIBRATION & UNCERTAINTY
# ================================================================
# Apply probability calibration and conformal prediction to the
# selected model's out-of-fold predictions. Every molecule gets
# both a calibrated probability AND an uncertainty band.
#
# Input:  Datasets/data/model_predictions_v1.parquet  (from Phase 4)
#         Datasets/data/features_combined.parquet
#         experiment/phase4_model_training/models/
# Output: Datasets/data/model_predictions.parquet   (final artifact)
#
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# !pip install mapie scikit-learn pandas pyarrow numpy matplotlib


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
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
PHASE5_DIR   = PROJECT_ROOT / "experiment" / "phase5_calibration_uncertainty"
MODEL_DIR    = PROJECT_ROOT / "experiment" / "phase4_model_training" / "models"

# ── Input ──────────────────────────────────────────────────
INPUT_PREDICTIONS = DATA_DIR / "model_predictions_v1.parquet"
INPUT_FEATURES    = DATA_DIR / "features_combined.parquet"

# ── Output (final project artifact) ───────────────────────
OUTPUT_PREDICTIONS = DATA_DIR / "model_predictions.parquet"
CALIBRATION_LOG    = PHASE5_DIR / "calibration_log.json"

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED  = 42
ALPHA        = 0.10    # conformal significance level (90% coverage)
N_BINS_CALIB = 10      # bins for calibration curve

np.random.seed(RANDOM_SEED)

# ── Verify ─────────────────────────────────────────────────
assert INPUT_PREDICTIONS.exists(), f"❌ {INPUT_PREDICTIONS} — Run Phase 4 first!"
assert INPUT_FEATURES.exists(), f"❌ {INPUT_FEATURES} — Run Phase 3 first!"
PHASE5_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 5 — CALIBRATION & UNCERTAINTY")
print("=" * 60)


# ================================================================
# Cell 2 — Load OOF Predictions
# ================================================================

df_preds = pd.read_parquet(INPUT_PREDICTIONS)
df_features = pd.read_parquet(INPUT_FEATURES)

selected_model = df_preds["selected_model"].iloc[0]

# Raw OOF predictions from the winning model
raw_col = "p_repellent_winner_raw"
y_true = df_preds["repellent_active"].values.astype(int)
y_raw  = df_preds[raw_col].values

print(f"\n✅ Loaded {len(df_preds)} OOF predictions.")
print(f"   Selected model: {selected_model}")
print(f"   Raw predictions range: [{y_raw.min():.4f}, {y_raw.max():.4f}]")


# ================================================================
# Cell 3 — Pre-Calibration Diagnostics
# ================================================================
# Evaluate calibration BEFORE applying any correction.

print("\n" + "=" * 60)
print("  PRE-CALIBRATION DIAGNOSTICS")
print("=" * 60)

# Brier score (lower is better)
brier_before = brier_score_loss(y_true, y_raw)
print(f"  Brier score (raw): {brier_before:.6f}")

# Calibration curve
prob_true, prob_pred = calibration_curve(y_true, y_raw, n_bins=N_BINS_CALIB, strategy="uniform")

# Expected Calibration Error (ECE)
bin_counts = np.histogram(y_raw, bins=N_BINS_CALIB, range=(0, 1))[0]
total = len(y_raw)
ece = np.sum(np.abs(prob_true - prob_pred) * bin_counts[:len(prob_true)] / total)
print(f"  ECE (raw): {ece:.6f}")

# Display calibration table
print(f"\n  {'Predicted':>10s}  {'Actual':>8s}  {'Count':>6s}")
print(f"  {'─'*10}  {'─'*8}  {'─'*6}")
for pp, pt, cnt in zip(prob_pred, prob_true, bin_counts):
    print(f"  {pp:>10.3f}  {pt:>8.3f}  {cnt:>6d}")

# Save pre-calibration reliability plot
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
ax.plot(prob_pred, prob_true, "o-", color="#4A90D9", label=f"{selected_model} (raw)")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Reliability Diagram (Pre-Calibration)")
ax.legend(loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
fig.savefig(PHASE5_DIR / "reliability_plot_pre_calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  📊 Saved: reliability_plot_pre_calibration.png")


# ================================================================
# Cell 4 — Isotonic Calibration (Fold-Aware)
# ================================================================
# Fit isotonic regression on OOF predictions to produce calibrated
# probabilities. We use a leave-one-fold-out approach to avoid
# overfitting the calibrator to the same predictions it will adjust.

print("\n" + "=" * 60)
print("  ISOTONIC CALIBRATION")
print("=" * 60)

fold_ids = df_preds["fold_id"].values.astype(int)
calibrated_probs = np.zeros_like(y_raw)

# Nested leave-one-fold-out: for each fold's OOF predictions,
# calibrate using the OTHER folds' OOF predictions.
unique_folds = np.unique(fold_ids)

for calib_fold in unique_folds:
    calib_mask = fold_ids == calib_fold
    train_mask = ~calib_mask

    # Fit isotonic on the training folds' OOF predictions
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_reg.fit(y_raw[train_mask], y_true[train_mask])

    # Apply to the held-out fold
    calibrated_probs[calib_mask] = iso_reg.transform(y_raw[calib_mask])

# Clip to (0,1) for safety
calibrated_probs = np.clip(calibrated_probs, 1e-8, 1 - 1e-8)

# Post-calibration metrics
brier_after = brier_score_loss(y_true, calibrated_probs)
prob_true_cal, prob_pred_cal = calibration_curve(y_true, calibrated_probs,
                                                  n_bins=N_BINS_CALIB, strategy="uniform")
bin_counts_cal = np.histogram(calibrated_probs, bins=N_BINS_CALIB, range=(0, 1))[0]
ece_after = np.sum(
    np.abs(prob_true_cal - prob_pred_cal) * bin_counts_cal[:len(prob_true_cal)] / total
)

print(f"  Brier score: {brier_before:.6f} → {brier_after:.6f}  "
      f"({'✅ improved' if brier_after < brier_before else '⚠️ worse'})")
print(f"  ECE:         {ece:.6f} → {ece_after:.6f}  "
      f"({'✅ improved' if ece_after < ece else '⚠️ worse'})")

# Save post-calibration reliability plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].plot(prob_pred, prob_true, "o-", color="#E74C3C", label="Raw")
axes[0].set_title("Before Calibration")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

# After
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].plot(prob_pred_cal, prob_true_cal, "o-", color="#27AE60", label="Calibrated")
axes[1].set_title("After Calibration")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)

fig.suptitle("Reliability Diagrams", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PHASE5_DIR / "reliability_plot_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  📊 Saved: reliability_plot_comparison.png")


# ================================================================
# Cell 5 — Conformal Prediction Intervals
# ================================================================
# Apply split conformal prediction to produce prediction intervals.
# Each molecule gets [lower, upper] bounds at the specified
# coverage level (1 - ALPHA = 90%).

print("\n" + "=" * 60)
print(f"  CONFORMAL PREDICTION (coverage = {(1 - ALPHA)*100:.0f}%)")
print("=" * 60)

# Conformal prediction for classification:
# For each calibration fold, compute nonconformity scores on the
# calibration data, then use those to construct prediction sets.

# Nonconformity score: |y - p_calibrated|
# For binary classification, we use a simpler approach:
# prediction interval on the probability axis.

conformal_lower = np.zeros_like(calibrated_probs)
conformal_upper = np.zeros_like(calibrated_probs)

for calib_fold in unique_folds:
    calib_mask = fold_ids == calib_fold
    train_mask = ~calib_mask

    # Compute nonconformity scores on calibration data
    # Score = absolute residual between prediction and true label
    scores = np.abs(y_true[train_mask] - calibrated_probs[train_mask])

    # Quantile of scores (with finite-sample correction)
    n_cal = scores.shape[0]
    q_level = np.ceil((1 - ALPHA) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(scores, q_level)

    # Prediction interval for held-out fold
    conformal_lower[calib_mask] = np.clip(calibrated_probs[calib_mask] - q_hat, 0, 1)
    conformal_upper[calib_mask] = np.clip(calibrated_probs[calib_mask] + q_hat, 0, 1)

# Interval width statistics
interval_widths = conformal_upper - conformal_lower
print(f"  Interval width: mean={interval_widths.mean():.4f}, "
      f"min={interval_widths.min():.4f}, max={interval_widths.max():.4f}")

# Coverage check
covered = ((y_true >= conformal_lower) & (y_true <= conformal_upper)).mean()
print(f"  Empirical coverage: {covered*100:.1f}% (target: {(1-ALPHA)*100:.0f}%)")


# ================================================================
# Cell 6 — OOD (Out-of-Distribution) Score
# ================================================================
# Compute an OOD score for each molecule based on its distance
# to the training data in feature space. Higher score = more
# out-of-distribution.

print("\n" + "=" * 60)
print("  OOD SCORING")
print("=" * 60)

# Get feature columns
meta_cols = [
    "compound_id", "canonical_smiles", "source_dataset",
    "repellent_active", "scaffold_smiles", "fold_id", "qc_status",
]
feature_cols = [c for c in df_features.columns if c not in meta_cols]

# Labeled feature matrix
df_labeled_features = df_features[df_features["repellent_active"].notna()]
X_labeled = df_labeled_features[feature_cols].values

# Impute and scale
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X_labeled)
sc = StandardScaler()
X_std = sc.fit_transform(X_imp)

# Fit k-NN on labeled data
K_OOD = 5
nn_model = NearestNeighbors(n_neighbors=K_OOD, metric="euclidean", n_jobs=-1)
nn_model.fit(X_std)

# OOD = mean distance to K nearest neighbors
distances, _ = nn_model.kneighbors(X_std)
ood_scores = distances.mean(axis=1)

# Normalize to [0, 1] range
ood_min, ood_max = ood_scores.min(), ood_scores.max()
if ood_max > ood_min:
    ood_scores_norm = (ood_scores - ood_min) / (ood_max - ood_min)
else:
    ood_scores_norm = np.zeros_like(ood_scores)

print(f"  OOD scores computed for {len(ood_scores)} labeled molecules.")
print(f"  Raw range:  [{ood_scores.min():.4f}, {ood_scores.max():.4f}]")
print(f"  Norm range: [{ood_scores_norm.min():.4f}, {ood_scores_norm.max():.4f}]")


# ================================================================
# Cell 7 — Assemble Final Prediction Table
# ================================================================
# Build the required output format per the project plan.

print("\n" + "=" * 60)
print("  ASSEMBLING FINAL PREDICTIONS")
print("=" * 60)

# Start with the labeled prediction dataframe
df_final = df_preds[["compound_id", "canonical_smiles",
                      "scaffold_smiles", "repellent_active",
                      "fold_id"]].copy()

# Rename fold_id to split_id for consistency with spec
df_final = df_final.rename(columns={"fold_id": "split_id"})

# Add calibrated probability
df_final["p_repellent"]     = calibrated_probs
df_final["p_non_repellent"] = 1.0 - calibrated_probs

# Add prediction interval
df_final["prediction_interval_lower"] = conformal_lower
df_final["prediction_interval_upper"] = conformal_upper
# Compact format as string
df_final["prediction_interval"] = [
    f"[{lo:.4f}, {up:.4f}]"
    for lo, up in zip(conformal_lower, conformal_upper)
]

# Add OOD score
df_final["ood_score"] = ood_scores_norm

# Add model info
df_final["model"] = selected_model
df_final["calibration_method"] = "isotonic_fold_aware"

# Reorder columns per project spec
column_order = [
    "compound_id",
    "canonical_smiles",
    "scaffold_smiles",       # scaffold_id in spec
    "p_repellent",
    "p_non_repellent",
    "prediction_interval",
    "prediction_interval_lower",
    "prediction_interval_upper",
    "ood_score",
    "split_id",
    "repellent_active",
    "model",
    "calibration_method",
]
df_final = df_final[column_order]

print(f"  Final prediction table: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
print(f"  Columns: {list(df_final.columns)}")


# ================================================================
# Cell 8 — Post-Calibration Metrics
# ================================================================
# Final evaluation with calibrated probabilities.

print("\n" + "=" * 60)
print("  POST-CALIBRATION METRICS")
print("=" * 60)

final_metrics = {
    "model":         selected_model,
    "roc_auc":       float(roc_auc_score(y_true, calibrated_probs)),
    "pr_auc":        float(average_precision_score(y_true, calibrated_probs)),
    "mcc":           float(matthews_corrcoef(y_true, (calibrated_probs >= 0.5).astype(int))),
    "brier_raw":     float(brier_before),
    "brier_calib":   float(brier_after),
    "ece_raw":       float(ece),
    "ece_calib":     float(ece_after),
    "coverage":      float(covered),
    "target_coverage": float(1 - ALPHA),
    "mean_interval_width": float(interval_widths.mean()),
}

for key, val in final_metrics.items():
    if isinstance(val, float):
        print(f"  {key:25s}: {val:.6f}")
    else:
        print(f"  {key:25s}: {val}")


# ================================================================
# Cell 9 — Save Final Predictions
# ================================================================

df_final.to_parquet(OUTPUT_PREDICTIONS, index=False, engine="pyarrow")
print(f"\n✅ Saved final predictions: {OUTPUT_PREDICTIONS}")
print(f"   Size: {OUTPUT_PREDICTIONS.stat().st_size / 1024:.1f} KB")


# ================================================================
# Cell 10 — Save Calibration Log
# ================================================================

calibration_log = {
    "pipeline":           "Phase 5 — Calibration & Uncertainty",
    "random_seed":        RANDOM_SEED,
    "calibration_method": "isotonic_fold_aware",
    "conformal_alpha":    ALPHA,
    "conformal_coverage": float(1 - ALPHA),
    "ood_method":         f"kNN (K={K_OOD}, euclidean)",
    "metrics":            final_metrics,
    "output_file":        str(OUTPUT_PREDICTIONS),
}

with open(CALIBRATION_LOG, "w") as f:
    json.dump(calibration_log, f, indent=2)

print(f"✅ Calibration log saved: {CALIBRATION_LOG}")


# ================================================================
# Cell 11 — Distribution Plots
# ================================================================
# Save plots of calibrated probability distributions by class.

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# (1) Calibrated probability distribution
axes[0].hist(calibrated_probs[y_true == 1], bins=30, alpha=0.6,
             color="#27AE60", label="Repellent (y=1)", density=True)
axes[0].hist(calibrated_probs[y_true == 0], bins=30, alpha=0.6,
             color="#E74C3C", label="Non-repellent (y=0)", density=True)
axes[0].set_xlabel("Calibrated p(repellent)")
axes[0].set_ylabel("Density")
axes[0].set_title("Calibrated Probability Distribution")
axes[0].legend()

# (2) OOD score distribution
axes[1].hist(ood_scores_norm[y_true == 1], bins=30, alpha=0.6,
             color="#27AE60", label="Repellent", density=True)
axes[1].hist(ood_scores_norm[y_true == 0], bins=30, alpha=0.6,
             color="#E74C3C", label="Non-repellent", density=True)
axes[1].set_xlabel("OOD Score (normalized)")
axes[1].set_ylabel("Density")
axes[1].set_title("OOD Score Distribution")
axes[1].legend()

# (3) Interval width distribution
axes[2].hist(interval_widths, bins=30, alpha=0.7, color="#4A90D9")
axes[2].set_xlabel("Prediction Interval Width")
axes[2].set_ylabel("Count")
axes[2].set_title(f"Conformal Interval Widths (α={ALPHA})")

fig.suptitle("Phase 5 — Calibration & Uncertainty Diagnostics", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(PHASE5_DIR / "calibration_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: calibration_diagnostics.png")


# ================================================================
# Cell 12 — Final Summary
# ================================================================

print("\n" + "=" * 60)
print("  PHASE 5 COMPLETE — Calibration & Uncertainty")
print("=" * 60)
print(f"\n  Key results:")
print(f"    • Calibration: Brier {brier_before:.4f} → {brier_after:.4f}")
print(f"    • Calibration: ECE {ece:.4f} → {ece_after:.4f}")
print(f"    • Conformal coverage: {covered*100:.1f}% (target: {(1-ALPHA)*100:.0f}%)")
print(f"    • Mean interval width: {interval_widths.mean():.4f}")
print(f"\n  Artifacts:")
print(f"    1. {OUTPUT_PREDICTIONS} (FINAL PROJECT ARTIFACT)")
print(f"    2. {CALIBRATION_LOG}")
print(f"    3. reliability_plot_comparison.png")
print(f"    4. calibration_diagnostics.png")
print(f"\n  Ready for Phase 6 (Interpretability) →")
print("=" * 60)

