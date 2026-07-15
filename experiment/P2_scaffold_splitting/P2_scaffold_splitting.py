# *PHASE 2 — SCAFFOLD-AWARE DATA SPLITTING*

# Creates scaffold-aware 5-fold cross-validation splits where
# NO Bemis–Murcko scaffold appears in both train and validation
# within any fold. This prevents structural data leakage.
# Handles the imbalanced repellency dataset (~411 pos / ~1,210 neg).

# # Input:
# 1. G:/research/ECOAI/experiment/P1_data_curation/artifacts/curated_molecules.parquet

# # Output:
# 1. G:/research/ECOAI/experiment/P2_scaffold_splitting/artifacts/split_manifest.json
# 2. G:/research/ECOAI/experiment/P2_scaffold_splitting/artifacts/curated_molecules_with_splits.parquet
# 3. G:/research/ECOAI/experiment/P2_scaffold_splitting/artifacts/split_statistics.json
# 4. G:/research/ECOAI/experiment/P2_scaffold_splitting/plots/P2_fold_distribution.png
# 5. G:/research/ECOAI/experiment/P2_scaffold_splitting/plots/P2_fold_distribution.jpg


# ================================================================
# S1 — Install Dependencies
# ================================================================
# Uncomment the line below and run ONCE to install all packages.
# !pip install pandas pyarrow numpy scikit-learn tqdm matplotlib


# ================================================================
# S2 — Imports and Configuration
# ================================================================
import json
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths
PROJECT_ROOT = Path(r"G:\research\ECOAI")
P1_ARTIFACTS = PROJECT_ROOT / "experiment" / "P1_data_curation" / "artifacts"
P2_DIR       = PROJECT_ROOT / "experiment" / "P2_scaffold_splitting"
ARTIFACTS_DIR = P2_DIR / "artifacts"
PLOTS_DIR     = P2_DIR / "plots"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input / Output
INPUT_PARQUET    = P1_ARTIFACTS / "curated_molecules.parquet"
OUTPUT_MANIFEST  = ARTIFACTS_DIR / "split_manifest.json"
OUTPUT_PARQUET   = ARTIFACTS_DIR / "curated_molecules_with_splits.parquet"
SPLIT_STATS_FILE = ARTIFACTS_DIR / "split_statistics.json"

N_FOLDS     = 5
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

assert INPUT_PARQUET.exists(), f"[ERROR] Input not found: {INPUT_PARQUET}"

print("=" * 60)
print("  P2 — SCAFFOLD-AWARE SPLITTING")
print("=" * 60)
print(f"  Input:  {INPUT_PARQUET}")
print(f"  Folds:  {N_FOLDS}")
print(f"  Seed:   {RANDOM_SEED}")


# ================================================================
# S3 — Load Curated Data
# ================================================================
df = pd.read_parquet(INPUT_PARQUET)
print(f"\n[OK] Loaded {len(df)} molecules from curated parquet.")

# Normalize empty scaffolds to NaN
df.loc[df["scaffold_smiles"] == "", "scaffold_smiles"] = np.nan
print("   Normalized empty scaffolds -> NaN")

# Filter to labeled + QC-pass molecules
df_labeled = df[
    (df["qc_status"] == "pass") &
    (df["repellent_active"].notna())
].copy()

print(f"   Labeled + QC-pass molecules: {len(df_labeled)}")
print(f"     repellent_active = 1: {(df_labeled['repellent_active'] == 1).sum()}")
print(f"     repellent_active = 0: {(df_labeled['repellent_active'] == 0).sum()}")
n_pos_total = (df_labeled['repellent_active'] == 1).sum()
n_neg_total = (df_labeled['repellent_active'] == 0).sum()
imbalance_ratio = n_neg_total / n_pos_total if n_pos_total > 0 else float('inf')
print(f"     Imbalance ratio:      {imbalance_ratio:.2f}:1 (neg:pos)")
if imbalance_ratio > 2:
    print(f"     [WARN] Significant class imbalance detected.")
print(f"   Unique scaffolds (excl. NaN): {df_labeled['scaffold_smiles'].nunique()}")


# ================================================================
# S4 — Scaffold-Aware Fold Assignment
# ================================================================
def scaffold_split_kfold(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold_smiles",
    n_folds: int = 5,
    seed: int = 42,
) -> pd.Series:
    """Assign fold IDs greedily to balance sizes while keeping scaffolds grouped."""
    rng = np.random.RandomState(seed)
    fold_ids = pd.Series(index=df.index, dtype=int)

    scaffold_groups = []
    no_scaffold_indices = []

    for scaffold, group in df.groupby(scaffold_col, dropna=False):
        indices = group.index.tolist()
        if pd.isna(scaffold) or scaffold is None or scaffold == "":
            no_scaffold_indices.extend(indices)
        else:
            scaffold_groups.append((scaffold, indices))

    scaffold_groups.sort(key=lambda x: len(x[1]), reverse=True)

    size_groups = defaultdict(list)
    for scaffold, indices in scaffold_groups:
        size_groups[len(indices)].append((scaffold, indices))

    shuffled_groups = []
    for size in sorted(size_groups.keys(), reverse=True):
        groups = size_groups[size]
        rng.shuffle(groups)
        shuffled_groups.extend(groups)

    fold_sizes = np.zeros(n_folds, dtype=int)

    for scaffold, indices in shuffled_groups:
        target_fold = int(np.argmin(fold_sizes))
        fold_ids.loc[indices] = target_fold
        fold_sizes[target_fold] += len(indices)

    rng.shuffle(no_scaffold_indices)
    for idx in no_scaffold_indices:
        target_fold = int(np.argmin(fold_sizes))
        fold_ids.loc[idx] = target_fold
        fold_sizes[target_fold] += 1

    return fold_ids

df_labeled["fold_id"] = scaffold_split_kfold(
    df_labeled,
    scaffold_col="scaffold_smiles",
    n_folds=N_FOLDS,
    seed=RANDOM_SEED,
)

print(f"\n[OK] Scaffold-aware {N_FOLDS}-fold split assigned.")


# ================================================================
# S5 — Verify Zero Scaffold Leakage
# ================================================================
print("\n" + "=" * 60)
print("  LEAKAGE VERIFICATION")
print("=" * 60)

all_clean = True
for fold in range(N_FOLDS):
    val_mask   = df_labeled["fold_id"] == fold
    train_mask = ~val_mask

    val_sc   = df_labeled.loc[val_mask, "scaffold_smiles"].dropna()
    train_sc = df_labeled.loc[train_mask, "scaffold_smiles"].dropna()
    val_scaffolds   = set(val_sc[val_sc != ""].unique())
    train_scaffolds = set(train_sc[train_sc != ""].unique())

    overlap = val_scaffolds & train_scaffolds
    n_train = train_mask.sum()
    n_val   = val_mask.sum()

    if len(overlap) == 0:
        print(f"  Fold {fold}: [OK] No leakage  (train={n_train}, val={n_val})")
    else:
        print(f"  Fold {fold}: [ERROR] LEAKAGE! {len(overlap)} shared scaffolds!")
        all_clean = False

if all_clean:
    print("\n  [OK] ALL FOLDS PASS — zero scaffold leakage confirmed!")
else:
    print("\n  [WARNING] LEAKAGE DETECTED — review the splitting algorithm!")


# ================================================================
# S6 — Fold Statistics
# ================================================================
print("\n" + "=" * 60)
print("  FOLD STATISTICS")
print("=" * 60)

fold_stats = {}
for fold in range(N_FOLDS):
    val_mask   = df_labeled["fold_id"] == fold
    train_mask = ~val_mask

    val_df   = df_labeled[val_mask]
    train_df = df_labeled[train_mask]

    fold_info = {
        "train_total":        int(train_mask.sum()),
        "train_positive":     int((train_df["repellent_active"] == 1).sum()),
        "train_negative":     int((train_df["repellent_active"] == 0).sum()),
        "train_scaffolds":    int(train_df["scaffold_smiles"].nunique()),
        "val_total":          int(val_mask.sum()),
        "val_positive":       int((val_df["repellent_active"] == 1).sum()),
        "val_negative":       int((val_df["repellent_active"] == 0).sum()),
        "val_scaffolds":      int(val_df["scaffold_smiles"].nunique()),
    }
    fold_stats[f"fold_{fold}"] = fold_info

    pos_rate_train = fold_info["train_positive"] / fold_info["train_total"] * 100
    pos_rate_val   = fold_info["val_positive"]   / fold_info["val_total"]   * 100

    print(f"\n  --- Fold {fold} ---")
    print(f"    Train: {fold_info['train_total']:>4d} mols  "
          f"({fold_info['train_positive']} pos / {fold_info['train_negative']} neg, "
          f"{pos_rate_train:.1f}% positive, "
          f"{fold_info['train_scaffolds']} scaffolds)")
    print(f"    Val:   {fold_info['val_total']:>4d} mols  "
          f"({fold_info['val_positive']} pos / {fold_info['val_negative']} neg, "
          f"{pos_rate_val:.1f}% positive, "
          f"{fold_info['val_scaffolds']} scaffolds)")


# ================================================================
# S7 — Build and Save Split Manifest
# ================================================================
manifest = {
    "description": "Scaffold-aware 5-fold CV split for repellency modeling",
    "n_folds": N_FOLDS,
    "random_seed": RANDOM_SEED,
    "split_method": "greedy_balanced_scaffold_split",
    "leakage_verified": all_clean,
    "total_labeled_molecules": len(df_labeled),
    "fold_assignments": {},
    "fold_statistics": fold_stats,
}

for _, row in df_labeled.iterrows():
    manifest["fold_assignments"][row["compound_id"]] = int(row["fold_id"])

with open(OUTPUT_MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n[OK] Split manifest saved: {OUTPUT_MANIFEST}")


# ================================================================
# S8 — Merge Fold IDs Back into Full Dataset
# ================================================================
df["fold_id"] = -1

fold_map = df_labeled[["compound_id", "fold_id"]].set_index("compound_id")["fold_id"]
df.loc[df["compound_id"].isin(fold_map.index), "fold_id"] = (
    df.loc[df["compound_id"].isin(fold_map.index), "compound_id"].map(fold_map).values
)

df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
print(f"\n[OK] Enriched parquet saved: {OUTPUT_PARQUET}")


# ================================================================
# S9 — Save Split Statistics
# ================================================================
split_statistics = {
    "n_folds": N_FOLDS,
    "random_seed": RANDOM_SEED,
    "leakage_verified": all_clean,
    "fold_stats": fold_stats,
    "label_distribution": {
        "total_labeled": len(df_labeled),
        "positive": int((df_labeled["repellent_active"] == 1).sum()),
        "negative": int((df_labeled["repellent_active"] == 0).sum()),
    },
    "scaffold_count": int(df_labeled["scaffold_smiles"].nunique()),
}

with open(SPLIT_STATS_FILE, "w") as f:
    json.dump(split_statistics, f, indent=2)

print(f"[OK] Split statistics saved: {SPLIT_STATS_FILE}")


# ================================================================
# S10 — Generate Fold Distribution Plot (PNG + JPG)
# ================================================================
folds = [f"Fold {i}" for i in range(N_FOLDS)]
pos_counts = [fold_stats[f"fold_{i}"]["val_positive"] for i in range(N_FOLDS)]
neg_counts = [fold_stats[f"fold_{i}"]["val_negative"] for i in range(N_FOLDS)]

x = np.arange(len(folds))
width = 0.35

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 12
})

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, pos_counts, width, label="Active Repellent (1)", color="#17a589", edgecolor="black", alpha=0.85)
bars2 = ax.bar(x + width/2, neg_counts, width, label="Inactive/Decoy (0)", color="#1b4f72", edgecolor="black", alpha=0.85)

ax.set_title("Validation Set Class Distribution across Scaffold Folds", fontsize=14, fontweight="bold", color="black")
ax.set_xlabel("Folds", fontsize=12, fontweight="bold", color="black")
ax.set_ylabel("Number of Molecules", fontsize=12, fontweight="bold", color="black")
ax.set_xticks(x)

# Format ticks
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
    label.set_color("black")
    label.set_fontsize(12)

for label in ax.get_yticklabels():
    label.set_fontweight("bold")
    label.set_color("black")
    label.set_fontsize(12)

legend = ax.legend(fontsize=11, frameon=True)
for text in legend.get_texts():
    text.set_fontweight("bold")
    text.set_color("black")

ax.grid(axis="y", linestyle="--", alpha=0.5)

# Value labels on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                str(int(h)), ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")

plt.tight_layout()

plot_path_png = PLOTS_DIR / "P2_fold_distribution.png"
plot_path_jpg = PLOTS_DIR / "P2_fold_distribution.jpg"

plt.savefig(plot_path_png, dpi=300)
plt.savefig(plot_path_jpg, dpi=300)
plt.close()

print(f"[OK] Saved plots:\n  - {plot_path_png}\n  - {plot_path_jpg}")


# ================================================================
# S11 — Final Verification Summary
# ================================================================
print("\n" + "=" * 60)
print("  P2 COMPLETE — Scaffold-Aware Splitting")
print("=" * 60)
print(f"  Artifact Directory: {ARTIFACTS_DIR}")
print(f"  Plots Directory:    {PLOTS_DIR}")
print()
print(f"  Artifacts Generated:")
print(f"    1. {OUTPUT_MANIFEST.name} (Fold assignment mapping)")
print(f"    2. {OUTPUT_PARQUET.name} (Enriched dataset with fold IDs)")
print(f"    3. {SPLIT_STATS_FILE.name} (Splitting statistics)")
print(f"    4. P2_fold_distribution.png / .jpg (Fold balance visualization)")
print()
print(f"  Summary:")
print(f"    - Scaffold folds:       {N_FOLDS}")
print(f"    - Total labeled split:  {len(df_labeled)}")
print(f"    - Unique scaffolds:     {df_labeled['scaffold_smiles'].nunique()}")
print(f"    - Zero leakage:         {'[OK] YES' if all_clean else '[ERROR] NO'}")
print("=" * 60)
