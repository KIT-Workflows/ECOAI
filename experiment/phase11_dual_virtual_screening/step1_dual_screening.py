# ================================================================
# PHASE 11 — DUAL VIRTUAL SCREENING (JUPYTER FORMAT)
# ================================================================
# Rank the entire ~2,880 LifeChemicals library using both the
# Repellency Model and the Insecticidal Model simultaneously.
# Identify the ultimate "Holy Grail" molecules: Safe & Effective.
#
# Process:
# 1. Load massive unlabelled library.
# 2. Vectorize through both neural/classical networks (10 Fold Models).
# 3. Apply calibrated bounds.
# 4. Filter and Export categorical Top lists and stunning visualizations.
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
import matplotlib.patches as patches

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT     = Path(r"G:\research\ECOAI")
DATA_DIR         = PROJECT_ROOT / "Datasets" / "data"

PHASE11_DIR      = PROJECT_ROOT / "experiment" / "phase11_dual_virtual_screening"
ARTIFACT_DIR     = PHASE11_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 4 (Repellency) Paths
REPEL_MODEL_DIR  = PROJECT_ROOT / "experiment" / "phase4_model_training" / "models"
REPEL_OOF_FILE   = DATA_DIR / "model_predictions_v1.parquet"
REPEL_FEAT_FILE  = DATA_DIR / "features_combined.parquet"

# Phase 10 (Insecticidal) Paths
INSECT_ARTIFACTS = DATA_DIR / "phase10_artifacts"
INSECT_MODEL_DIR = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation" / "models"
INSECT_OOF_FILE  = INSECT_ARTIFACTS / "classical_oof_preds.parquet"
INSECT_META_FILE = INSECT_ARTIFACTS / "insecticide_meta.parquet"

print("=" * 60)
print("  PHASE 11 — DUAL VIRTUAL SCREENING")
print(f"  Artifacts will be saved to: {ARTIFACT_DIR}")
print("=" * 60)


# ================================================================
# Cell 2 — Load Datasets and Extract Unlabelled LifeChemicals
# ================================================================

print("\n── Loading Unlabelled Library (This takes a moment) ──")
df_all = pd.read_parquet(REPEL_FEAT_FILE)

meta_cols = [
    "compound_id", "canonical_smiles", "source_dataset",
    "repellent_active", "scaffold_smiles", "fold_id", "qc_status"
]
feature_cols = [c for c in df_all.columns if c not in meta_cols]

# We only screen the unlabelled molecules!
df_screen = df_all[df_all["repellent_active"].isna()].copy()
df_screen = df_screen.reset_index(drop=True)

X_screen = df_screen[feature_cols].values

print(f"✅ Library Loaded!")
print(f"   Molecules to screen: {len(df_screen):,}")


# ================================================================
# Cell 3 — Load Models & Calibrators
# ================================================================

print("\n── Loading Models and Fitting Calibrators ──")

# 1. Repellency Models & Calibrator
models_repel = []
for fold in range(5):
    with open(REPEL_MODEL_DIR / f"lgb_fold_{fold}.pkl", "rb") as f:
        models_repel.append(pickle.load(f))

df_repel = pd.read_parquet(REPEL_OOF_FILE)
iso_repel = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso_repel.fit(df_repel["p_repellent_winner_raw"], df_repel["repellent_active"])

res_r = np.abs(df_repel["repellent_active"] - iso_repel.transform(df_repel["p_repellent_winner_raw"]))
q_hat_r = np.quantile(res_r, 0.90)

# 2. Insecticidal Models & Calibrator
models_insect = []
for fold in range(5):
    with open(INSECT_MODEL_DIR / f"classical_fold_{fold}.pkl", "rb") as f:
        models_insect.append(pickle.load(f))

df_ins_oof = pd.read_parquet(INSECT_OOF_FILE)
df_ins_meta = pd.read_parquet(INSECT_META_FILE)
df_ins_merge = pd.merge(df_ins_oof, df_ins_meta, on="compound_id", how="inner")

iso_insect = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso_insect.fit(df_ins_merge["p_insecticidal_raw"], df_ins_merge["insecticidal_active"])

res_i = np.abs(df_ins_merge["insecticidal_active"] - iso_insect.transform(df_ins_merge["p_insecticidal_raw"]))
q_hat_i = np.quantile(res_i, 0.90)

print(f"✅ Loaded 10 Models (5 Repellency, 5 Insecticidal) & Calibrators Ready.")


# ================================================================
# Cell 4 — Massive Matrix Prediction
# ================================================================

print("\n" + "=" * 60)
print(f"  COMMENCING DUAL INFERENCE ON {len(X_screen):,} MOLECULES")
print("=" * 60)

# A. Repellency Inference
print("\n[1/2] Matrix Inference: Repellency Brain (LightGBM)")
repel_preds = []
for i, m in enumerate(models_repel):
    X_proc = m["scaler"].transform(m["imputer"].transform(X_screen))
    repel_preds.append(m["model"].predict_proba(X_proc)[:, 1])
    print(f"      Fold {i+1}/5 finished...")

r_raw = np.mean(repel_preds, axis=0)
p_repel_cal = iso_repel.transform(r_raw)


# B. Insecticidal Inference
print("\n[2/2] Matrix Inference: Insecticidal Brain (RandomForest)")
insect_preds = []
for i, m in enumerate(models_insect):
    X_proc = m["scaler"].transform(m["imputer"].transform(X_screen))
    insect_preds.append(m["model"].predict_proba(X_proc)[:, 1])
    print(f"      Fold {i+1}/5 finished...")

i_raw = np.mean(insect_preds, axis=0)
p_insect_cal = iso_insect.transform(i_raw)

print(f"✅ Massive Matrix Prediction Complete.")


# ================================================================
# Cell 5 — Build Ultimate Screening Dataset & Categories
# ================================================================

df_results = pd.DataFrame({
    "compound_id": df_screen["compound_id"],
    "canonical_smiles": df_screen["canonical_smiles"],
    "p_repellent": p_repel_cal,
    "p_insecticidal": p_insect_cal,
    "repel_lower_90": np.clip(p_repel_cal - q_hat_r, 0, 1),
    "repel_upper_90": np.clip(p_repel_cal + q_hat_r, 0, 1),
    "insect_lower_90": np.clip(p_insect_cal - q_hat_i, 0, 1),
    "insect_upper_90": np.clip(p_insect_cal + q_hat_i, 0, 1),
})

# Classify each molecule
def categorize(row):
    r, i = row["p_repellent"], row["p_insecticidal"]
    if r > 0.5 and i < 0.2:
        return "Holy Grail (Safe Repellent)"
    elif r > 0.5 and i >= 0.2:
        return "Toxic Repellent"
    elif r <= 0.5 and i >= 0.5:
        return "Bug Spray (Toxic)"
    else:
        return "Inactive"

df_results["Category"] = df_results.apply(categorize, axis=1)

counts = df_results["Category"].value_counts()
print("\n── Entire Library Classification Summary ──")
for cat, count in counts.items():
    print(f"  {cat:30s} : {count:,}")


# ================================================================
# Cell 6 — Export Artifacts (Full Library & Top 50s)
# ================================================================
print("\n── Exporting High-Value Data Artifacts ──")

# 1. Save Full Parquet (Everything)
FULL_PARQUET = ARTIFACT_DIR / "full_library_predictions.parquet"
df_results.to_parquet(FULL_PARQUET, index=False)
print(f"✅ Saved full library predictions to: {FULL_PARQUET.name}")

# 2. Extract specific Top Lists
def export_top(category_name, sort_cols, ascending_list, filename, n_max=50):
    subset = df_results[df_results["Category"] == category_name]
    if len(subset) == 0: return
    top_candidates = subset.sort_values(by=sort_cols, ascending=ascending_list).head(n_max)
    out_path = ARTIFACT_DIR / filename
    top_candidates.to_csv(out_path, index=False)
    print(f"✅ Saved top candidates for {category_name.split('(')[0].strip()} to: {out_path.name}")

# The Holy Grails (Maximize Repellency, Minimize Insecticidal)
export_top("Holy Grail (Safe Repellent)", ["p_repellent", "p_insecticidal"], [False, True], "top_holy_grails.csv")

# The Toxic Repellents (Useful for agricultural use, maximize both)
export_top("Toxic Repellent", ["p_repellent", "p_insecticidal"], [False, False], "top_toxic_repellents.csv")

# The Pure Bug Sprays (Minimize Repellency, Maximize Insecticidal)
export_top("Bug Spray (Toxic)", ["p_insecticidal", "p_repellent"], [False, True], "top_bug_sprays.csv")


# ================================================================
# Cell 7 — Visualizations
# ================================================================
print("\n── Rendering Visualizations ──")

# A. The Dual-Quadrant Map (Styled Consistently)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 10
})

fig, ax = plt.subplots(figsize=(7, 7)) 
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot Background Points (Inactive - Grey)
inactive = df_results[df_results["Category"] == "Inactive"]
ax.scatter(inactive["p_insecticidal"], inactive["p_repellent"], 
           alpha=0.4, s=6, c="#95A5A6", label="Inactive")

# Plot Toxic Repellents (Blue)
tox_rep = df_results[df_results["Category"] == "Toxic Repellent"]
ax.scatter(tox_rep["p_insecticidal"], tox_rep["p_repellent"], 
           alpha=0.6, s=15, c="#4A90D9", label="Toxic Repellent")

# Plot Bug Sprays (Red)
bug_spray = df_results[df_results["Category"] == "Bug Spray (Toxic)"]
ax.scatter(bug_spray["p_insecticidal"], bug_spray["p_repellent"], 
           alpha=0.6, s=15, c="#E74C3C", label="Bug Spray (Toxic)")

# Plot Holy Grails (Green)
grails = df_results[df_results["Category"] == "Holy Grail (Safe Repellent)"]
ax.scatter(grails["p_insecticidal"], grails["p_repellent"], 
           alpha=0.9, s=40, c="#27AE60", edgecolor="black", linewidth=0.8, label="Holy Grail (Safe Repellent)")

# Quadrant Lines
ax.axvline(0.2, color='black', linestyle='--', alpha=0.3)
ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)

# Aesthetics
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# Bold for consistency
ax.set_xlabel("Insecticidal Probability (Random Forest)", fontsize=10, fontweight="bold", color="black")
ax.set_ylabel("Repellent Probability (XGBoost)", fontsize=10, fontweight="bold", color="black")

# Titles are commented out for markdown consistency
# ax.set_title("Dual Virtual Screening: Efficacy vs. Toxicity Map", fontsize=14, fontweight="bold", color="black", pad=15)

# Ticks styling
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
    label.set_color("black")

# Grid
ax.grid(True, linestyle="--", alpha=0.3)

# Legend (Cleanly placed in a box above the plot)
legend = ax.legend(
    fontsize=9, 
    frameon=True,
    facecolor="white",
    edgecolor="#BDC3C7",
    loc="upper center", 
    bbox_to_anchor=(0.5, 1.15), 
    ncol=2
)
for text in legend.get_texts():
    text.set_fontweight("bold")

# Highlight box for Holy Grails
rect = patches.Rectangle((-0.02, 0.5), 0.22, 0.52, linewidth=1.5, edgecolor='#27AE60', facecolor='#27AE60', alpha=0.08)
ax.add_patch(rect)

for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.0)

fig.tight_layout()
quadrant_path = ARTIFACT_DIR / "xgb_rf_dual_screening_quadrant_map.png"
fig.savefig(quadrant_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"📊 Saved Map to: {quadrant_path.name}")


# B. Category Bar Chart (Styled Consistently)
cat_colors = {"Inactive": "#95A5A6", "Toxic Repellent": "#4A90D9", "Bug Spray (Toxic)": "#E74C3C", "Holy Grail (Safe Repellent)": "#27AE60"}
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bars = ax.bar(counts.index, counts.values, color=[cat_colors.get(x, "#333333") for x in counts.index], edgecolor="black", linewidth=1.2, width=0.6)

# Labels bold for consistency
ax.set_ylabel("Number of Molecules (Log Scale)", fontsize=10, fontweight="bold", color="black")
ax.set_yscale("log")
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Ticks
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
    label.set_color("black")

# Rotate x-axis labels to fit better
plt.xticks(rotation=20, ha="right")

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval * 1.15, f"{int(yval):,}", ha='center', va='bottom', fontsize=10, fontweight="bold", color="black")

plt.tight_layout()
bar_path = ARTIFACT_DIR / "xgb_rf_dual_screening_categories_bar.png"
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"📊 Saved Bar Chart to: {bar_path.name}")

print("\n" + "=" * 60)
print("  PHASE 11 ARTIFACT GENERATION COMPLETE!")
print("=" * 60)
