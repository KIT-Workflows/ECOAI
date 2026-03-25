# ================================================================
# STEP 7 — CALIBRATION & UNCERTAINTY (Insecticidal Pipeline)
# ================================================================
# Apply Isotonic Regression (fold-aware) on the winning model.
# Compute Conformal Prediction Intervals and k-NN OOD scores.
# Construct the final 4-level endpoint table.
#
# Input:  implementation/artifacts/winner_results.json
#         implementation/artifacts/classical_oof_preds.parquet
#         implementation/artifacts/ft_transformer_oof_preds.parquet
# Output: implementation/artifacts/insecticidal_predictions.parquet
#         implementation/artifacts/calibration_results.json
# ================================================================


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

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
IMPL_DIR     = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation"
ARTIFACT_DIR = DATA_DIR / "phase10_artifacts"

INPUT_META   = ARTIFACT_DIR / "insecticide_meta.parquet"
INPUT_FEAT   = ARTIFACT_DIR / "insecticide_features.npz"
INPUT_WINNER = ARTIFACT_DIR / "winner_results.json"

assert INPUT_WINNER.exists(), f"❌ {INPUT_WINNER} — Run step6 first!"

N_FOLDS = 5
ALPHA = 0.10
K_OOD = 5

print("=" * 65)
print("  STEP 7 — CALIBRATION & UNCERTAINTY")
print("=" * 65)


# ================================================================
# Cell 2 — Load Data & Winner Predictions
# ================================================================
with open(INPUT_WINNER, "r") as f:
    winner_res = json.load(f)

winner = winner_res["winner_name"]
print(f"\n✅ Winning model: {winner}")

df_meta = pd.read_parquet(INPUT_META)
y = df_meta["insecticidal_active"].values.astype(int)
fold_ids = df_meta["fold_id"].values.astype(int)

# Load winner OOF predictions
if winner == "FT-Transformer":
    pred_path = ARTIFACT_DIR / "ft_transformer_oof_preds.parquet"
else:
    pred_path = ARTIFACT_DIR / "classical_oof_preds.parquet"

df_preds = pd.read_parquet(pred_path)
winner_preds = df_preds["p_insecticidal_raw"].values

print(f"✅ Loaded raw predictions: {len(winner_preds)}")


# ================================================================
# Cell 3 — Isotonic Calibration (Fold-Aware)
# ================================================================
print("\n" + "=" * 65)
print("  ISOTONIC CALIBRATION (fold-aware)")
print("=" * 65)

calibrated_probs = np.zeros_like(winner_preds)
brier_before = brier_score_loss(y, winner_preds)

for cf in range(N_FOLDS):
    cm = (fold_ids == cf)
    tm = (fold_ids != cf)
    
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(winner_preds[tm], y[tm])
    calibrated_probs[cm] = iso.transform(winner_preds[cm])

calibrated_probs = np.clip(calibrated_probs, 1e-8, 1 - 1e-8)
brier_after = brier_score_loss(y, calibrated_probs)

N_BINS_CALIB = 10
pt_r, pp_r = calibration_curve(y, winner_preds, n_bins=N_BINS_CALIB, strategy="uniform")
pt_c, pp_c = calibration_curve(y, calibrated_probs, n_bins=N_BINS_CALIB, strategy="uniform")

bc_r = np.histogram(winner_preds, bins=N_BINS_CALIB, range=(0,1))[0]
bc_c = np.histogram(calibrated_probs, bins=N_BINS_CALIB, range=(0,1))[0]

total = len(y)
ece_before = np.sum(np.abs(pt_r - pp_r) * bc_r[:len(pt_r)] / total)
ece_after  = np.sum(np.abs(pt_c - pp_c) * bc_c[:len(pt_c)] / total)

print(f"  Brier: {brier_before:.6f} → {brier_after:.6f}  "
      f"({'✅ improved' if brier_after < brier_before else '⚠️ worse'})")
print(f"  ECE:   {ece_before:.6f} → {ece_after:.6f}  "
      f"({'✅ improved' if ece_after < ece_before else '⚠️ worse'})")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot([0,1],[0,1],"k--",alpha=0.5); axes[0].plot(pp_r, pt_r, "o-", color="#E74C3C")
axes[0].set_title("Before"); axes[0].grid(True, alpha=0.3)
axes[1].plot([0,1],[0,1],"k--",alpha=0.5); axes[1].plot(pp_c, pt_c, "o-", color="#27AE60")
axes[1].set_title("After"); axes[1].grid(True, alpha=0.3)
fig.suptitle(f"{winner} — Reliability Diagrams", fontsize=14, fontweight="bold")
fig.tight_layout()

rel_plot_path = ARTIFACT_DIR / "reliability_plot.png"
fig.savefig(rel_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  📊 Saved: {rel_plot_path}")


# ================================================================
# Cell 4 — Conformal Prediction Intervals
# ================================================================
print("\n" + "=" * 65)
print(f"  CONFORMAL PREDICTION (coverage = {(1-ALPHA)*100:.0f}%)")
print("=" * 65)

conf_lo = np.zeros_like(calibrated_probs)
conf_hi = np.zeros_like(calibrated_probs)

for cf in range(N_FOLDS):
    cm, tm = (fold_ids == cf), (fold_ids != cf)
    scores = np.abs(y[tm] - calibrated_probs[tm])
    nc = scores.shape[0]
    ql = min(np.ceil((1-ALPHA)*(nc+1))/nc, 1.0)
    qh = np.quantile(scores, ql)
    
    conf_lo[cm] = np.clip(calibrated_probs[cm] - qh, 0, 1)
    conf_hi[cm] = np.clip(calibrated_probs[cm] + qh, 0, 1)

iw = conf_hi - conf_lo
covered = ((y >= conf_lo) & (y <= conf_hi)).mean()

print(f"  Interval width: mean={iw.mean():.4f}, min={iw.min():.4f}, max={iw.max():.4f}")
print(f"  Empirical coverage: {covered*100:.1f}% (target: {(1-ALPHA)*100:.0f}%)")


# ================================================================
# Cell 5 — OOD Scoring
# ================================================================
print("\n" + "=" * 65)
print("  OOD SCORING")
print("=" * 65)

npz = np.load(INPUT_FEAT)
X_all = npz["X"]

imp_o = SimpleImputer(strategy="median"); X_i = imp_o.fit_transform(X_all)
sc_o = StandardScaler(); X_s = sc_o.fit_transform(X_i)

nn_m = NearestNeighbors(n_neighbors=K_OOD, metric="euclidean", n_jobs=-1)
nn_m.fit(X_s)
dists, _ = nn_m.kneighbors(X_s)

ood = dists.mean(axis=1)
ood_n = (ood - ood.min()) / (ood.max() - ood.min()) if ood.max() > ood.min() else np.zeros_like(ood)

print(f"  OOD range: [{ood.min():.4f}, {ood.max():.4f}]")


# ================================================================
# Cell 6 — Assemble Final Predictions
# ================================================================
print("\n" + "=" * 65)
print("  ASSEMBLING FINAL PREDICTIONS")
print("=" * 65)

df_final = pd.DataFrame({
    "compound_id": df_meta["compound_id"].values,
    "inchikey": df_meta["inchikey"].values,
    "canonical_smiles": df_meta["canonical_smiles"].values,
    "insecticidal_active": y,
    "p_insecticidal": calibrated_probs,
    "p_non_insecticidal": 1.0 - calibrated_probs,
    "prediction_interval_lower": conf_lo,
    "prediction_interval_upper": conf_hi,
    "prediction_interval": [f"[{lo:.4f}, {up:.4f}]" for lo, up in zip(conf_lo, conf_hi)],
    "ood_score": ood_n,
    "fold_id": fold_ids,
    "in_lifechem": df_meta["in_lifechem"].values,
    "model": winner,
    "calibration_method": "isotonic_fold_aware",
})

final_path = ARTIFACT_DIR / "insecticidal_predictions.parquet"
df_final.to_parquet(final_path, index=False, engine="pyarrow")

print(f"  ✅ Saved: {final_path}")

calib_stats = {
    "brier_raw": float(brier_before), "brier_cal": float(brier_after),
    "ece_raw": float(ece_before), "ece_cal": float(ece_after),
    "coverage": float(covered), "mean_iw": float(iw.mean())
}
with open(ARTIFACT_DIR / "calibration_results.json", "w") as f:
    json.dump(calib_stats, f, indent=4)

print(f"\n  → End of pipeline! Next: Phase 6 (interpretability) or Virtual Screening.")
print("=" * 65)
