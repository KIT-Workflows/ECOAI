# ================================================================
# STEP 6 — MODEL SELECTION (Insecticidal Pipeline)
# ================================================================
# Compare the best classical model against the FT-Transformer.
# Print full multi-model report, select the winner based on PR-AUC.
# Resolves ties using Brier score. Saves final winner results.
#
# Input:  implementation/artifacts/classical_results.json
#         implementation/artifacts/ft_transformer_results.json
# Output: implementation/artifacts/winner_results.json
# ================================================================


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
IMPL_DIR     = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation"
ARTIFACT_DIR = DATA_DIR / "phase10_artifacts"
MODEL_DIR    = IMPL_DIR / "models"

CLASSICAL_FILE = ARTIFACT_DIR / "classical_results.json"
FT_FILE        = ARTIFACT_DIR / "ft_transformer_results.json"

assert CLASSICAL_FILE.exists(), f"❌ {CLASSICAL_FILE} — Run step4 first!"
# FT_FILE might be missing if the user skipped step 5
has_ft = FT_FILE.exists()

print("=" * 65)
print("  STEP 6 — FINAL MODEL COMPARISON & SELECTION")
print("=" * 65)


# ================================================================
# Cell 2 — Load Results
# ================================================================
with open(CLASSICAL_FILE, "r") as f:
    classical_res = json.load(f)

best_classical_name = classical_res["best_classical_model"]
best_classical_metrics = classical_res["best_metrics"]
random_baseline = classical_res["random_baseline"]

print(f"✅ Loaded classical results. Best: {best_classical_name}")

if has_ft:
    with open(FT_FILE, "r") as f:
        ft_res = json.load(f)
    overall_ft = ft_res["overall"]
    print(f"✅ Loaded FT-Transformer results.")
else:
    print(f"⚠️  FT-Transformer results not found ({FT_FILE.name}).")
    print(f"   (Proceeding with classical models only.)")
    overall_ft = None


# ================================================================
# Cell 3 — Comparison Matrix
# ================================================================
print("\n" + "=" * 60)
print("  FULL MODEL LEADERBOARD")
print("=" * 60)

all_overalls = []
for m_name, m_data in classical_res["all_models"].items():
    all_overalls.append(m_data["overall"])

if has_ft:
    all_overalls.append(overall_ft)

comparison = pd.DataFrame(all_overalls).set_index("model")
comparison = comparison.sort_values("pr_auc", ascending=False)
print(comparison.to_string(float_format="{:.4f}".format))

print(f"\n  Random baseline PR-AUC: {random_baseline:.4f}")
for name, row in comparison.iterrows():
    gain = (row["pr_auc"] - random_baseline) / random_baseline * 100
    print(f"  {'✅' if gain>0 else '❌'} {name}: +{gain:.1f}% over random PR-AUC")


# ================================================================
# Cell 4 — Winner Selection
# ================================================================
print("\n" + "=" * 60)
print("  WINNER SELECTION")
print("=" * 60)

winner = best_classical_name
winner_metrics = best_classical_metrics

if has_ft:
    print(f"  Challenger (FT-Transformer) PR-AUC: {overall_ft['pr_auc']:.4f}")
    print(f"  Defender ({best_classical_name}) PR-AUC: {best_classical_metrics['pr_auc']:.4f}")
    
    if overall_ft["pr_auc"] > best_classical_metrics["pr_auc"]:
        winner = "FT-Transformer"
        winner_metrics = overall_ft
    elif overall_ft["pr_auc"] == best_classical_metrics["pr_auc"]:
        if overall_ft["brier"] < best_classical_metrics["brier"]:
            winner = "FT-Transformer"
            winner_metrics = overall_ft
            
print(f"\n  🏆 OVERALL WINNER: {winner.upper()}")
print(f"     PR-AUC:  {winner_metrics['pr_auc']:.4f}")
print(f"     ROC-AUC: {winner_metrics['roc_auc']:.4f}")
print(f"     Brier:   {winner_metrics['brier']:.4f}")


# ================================================================
# Cell 5 — Save Winner Manifest
# ================================================================
winner_results = {
    "winner_name": winner,
    "winner_metrics": winner_metrics,
    "best_classical_name": best_classical_name,
    "best_classical_metrics": best_classical_metrics,
    "ft_transformer_metrics": overall_ft,
    "random_baseline": random_baseline
}

winner_path = ARTIFACT_DIR / "winner_results.json"
with open(winner_path, "w") as f:
    json.dump(winner_results, f, indent=4)

print(f"\n✅ Saved winner details: {winner_path}")
print(f"\n  → Next: Run step7_calibration_uncertainty.py")
print("=" * 65)
