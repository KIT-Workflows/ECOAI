# ================================================================
# PHASE 2 — SCAFFOLD-AWARE DATA SPLITTING
# ================================================================
# Creates scaffold-aware 5-fold cross-validation splits where
# NO Bemis–Murcko scaffold appears in both train and validation
# within any fold. This prevents structural data leakage.
#
# Input:  Datasets/data/curated_molecules.parquet  (from Phase 1)
# Output: Datasets/data/split_manifest.json
#         Datasets/data/curated_molecules_with_splits.parquet
#
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# !pip install pandas pyarrow numpy scikit-learn tqdm


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
PHASE2_DIR   = PROJECT_ROOT / "experiment" / "phase2_scaffold_splitting"

# ── Input / Output ─────────────────────────────────────────
INPUT_PARQUET  = DATA_DIR / "curated_molecules.parquet"
OUTPUT_MANIFEST = DATA_DIR / "split_manifest.json"
OUTPUT_PARQUET  = DATA_DIR / "curated_molecules_with_splits.parquet"
SPLIT_STATS_FILE = PHASE2_DIR / "split_statistics.json"

# ── Constants ──────────────────────────────────────────────
N_FOLDS     = 5
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Verify ─────────────────────────────────────────────────
assert INPUT_PARQUET.exists(), f"❌ Input not found: {INPUT_PARQUET}\n   Run Phase 1 first!"
PHASE2_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 2 — SCAFFOLD-AWARE SPLITTING")
print("=" * 60)
print(f"  Input:  {INPUT_PARQUET}")
print(f"  Folds:  {N_FOLDS}")
print(f"  Seed:   {RANDOM_SEED}")


# ================================================================
# Cell 2 — Load Curated Data
# ================================================================

df = pd.read_parquet(INPUT_PARQUET)
print(f"\n✅ Loaded {len(df)} molecules from curated parquet.")

# ── Normalize empty scaffolds to NaN ──────────────────────
# Acyclic molecules (no rings) get an empty string "" from
# RDKit's Murcko decomposition. These have no shared structural
# core, so they should be treated as scaffold-less (NaN) and
# assigned independently to folds.
df.loc[df["scaffold_smiles"] == "", "scaffold_smiles"] = np.nan

n_empty_fixed = (df["scaffold_smiles"].isna()).sum()
print(f"   Normalized empty scaffolds → NaN")

# ── Filter to labeled + QC-pass molecules for splitting ────
df_labeled = df[
    (df["qc_status"] == "pass") &
    (df["repellent_active"].notna())
].copy()

print(f"   Labeled + QC-pass molecules: {len(df_labeled)}")
print(f"     repellent_active = 1: {(df_labeled['repellent_active'] == 1).sum()}")
print(f"     repellent_active = 0: {(df_labeled['repellent_active'] == 0).sum()}")
print(f"   Unique scaffolds (excl. NaN): {df_labeled['scaffold_smiles'].nunique()}")


# ================================================================
# Cell 3 — Scaffold-Aware Fold Assignment
# ================================================================
# Strategy: group molecules by scaffold, then assign entire
# scaffold groups to folds. This guarantees zero scaffold leakage.
#
# We use a greedy balanced assignment: sort scaffolds by size
# (descending), then assign each scaffold to the fold that
# currently has the fewest molecules. This keeps folds as
# balanced as possible.

def scaffold_split_kfold(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold_smiles",
    n_folds: int = 5,
    seed: int = 42,
) -> pd.Series:
    """
    Assign each row a fold_id (0..n_folds-1) such that all molecules
    sharing a scaffold land in the SAME fold.

    Algorithm:
        1. Group molecules by scaffold.
        2. Handle molecules with missing scaffolds: assign each
           individually to maintain balance.
        3. Sort scaffold groups by size (descending).
        4. Shuffle groups of equal size for randomness.
        5. Greedily assign each group to the fold with fewest
           molecules so far.

    Returns a Series of fold IDs aligned with df's index.
    """
    rng = np.random.RandomState(seed)
    fold_ids = pd.Series(index=df.index, dtype=int)

    # ── Group by scaffold ──
    scaffold_groups = []
    no_scaffold_indices = []

    for scaffold, group in df.groupby(scaffold_col, dropna=False):
        indices = group.index.tolist()
        if pd.isna(scaffold) or scaffold is None or scaffold == "":
            # Molecules without scaffolds: treat each individually
            no_scaffold_indices.extend(indices)
        else:
            scaffold_groups.append((scaffold, indices))

    # ── Sort by group size (descending), shuffle ties ──
    scaffold_groups.sort(key=lambda x: len(x[1]), reverse=True)

    # Group scaffolds by size and shuffle within each size group
    size_groups = defaultdict(list)
    for scaffold, indices in scaffold_groups:
        size_groups[len(indices)].append((scaffold, indices))

    shuffled_groups = []
    for size in sorted(size_groups.keys(), reverse=True):
        groups = size_groups[size]
        rng.shuffle(groups)
        shuffled_groups.extend(groups)

    # ── Greedy assignment: always put next group in smallest fold ──
    fold_sizes = np.zeros(n_folds, dtype=int)

    for scaffold, indices in shuffled_groups:
        # Pick the fold with the fewest molecules
        target_fold = int(np.argmin(fold_sizes))
        fold_ids.loc[indices] = target_fold
        fold_sizes[target_fold] += len(indices)

    # ── Assign no-scaffold molecules individually ──
    rng.shuffle(no_scaffold_indices)
    for idx in no_scaffold_indices:
        target_fold = int(np.argmin(fold_sizes))
        fold_ids.loc[idx] = target_fold
        fold_sizes[target_fold] += 1

    return fold_ids


# ── Run the scaffold split ─────────────────────────────────
df_labeled["fold_id"] = scaffold_split_kfold(
    df_labeled,
    scaffold_col="scaffold_smiles",
    n_folds=N_FOLDS,
    seed=RANDOM_SEED,
)

print(f"\n✅ Scaffold-aware {N_FOLDS}-fold split assigned.")


# ================================================================
# Cell 4 — Verify Zero Scaffold Leakage
# ================================================================
# Critical check: for every fold, verify that NO scaffold appears
# in both the train set and the validation set.

print("\n" + "=" * 60)
print("  LEAKAGE VERIFICATION")
print("=" * 60)

all_clean = True

for fold in range(N_FOLDS):
    val_mask   = df_labeled["fold_id"] == fold
    train_mask = ~val_mask

    # Filter out NaN AND empty strings for robust leakage check
    val_sc   = df_labeled.loc[val_mask, "scaffold_smiles"].dropna()
    train_sc = df_labeled.loc[train_mask, "scaffold_smiles"].dropna()
    val_scaffolds   = set(val_sc[val_sc != ""].unique())
    train_scaffolds = set(train_sc[train_sc != ""].unique())

    overlap = val_scaffolds & train_scaffolds

    n_train = train_mask.sum()
    n_val   = val_mask.sum()

    if len(overlap) == 0:
        print(f"  Fold {fold}: ✅ No leakage  (train={n_train}, val={n_val})")
    else:
        print(f"  Fold {fold}: ❌ LEAKAGE! {len(overlap)} shared scaffolds!")
        all_clean = False

if all_clean:
    print("\n  🎉 ALL FOLDS PASS — zero scaffold leakage confirmed!")
else:
    print("\n  ⚠️  LEAKAGE DETECTED — review the splitting algorithm!")


# ================================================================
# Cell 5 — Fold Statistics
# ================================================================
# Show detailed composition of each fold.

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

    print(f"\n  ── Fold {fold} ──")
    print(f"    Train: {fold_info['train_total']:>4d} mols  "
          f"({fold_info['train_positive']} pos / {fold_info['train_negative']} neg, "
          f"{pos_rate_train:.1f}% positive, "
          f"{fold_info['train_scaffolds']} scaffolds)")
    print(f"    Val:   {fold_info['val_total']:>4d} mols  "
          f"({fold_info['val_positive']} pos / {fold_info['val_negative']} neg, "
          f"{pos_rate_val:.1f}% positive, "
          f"{fold_info['val_scaffolds']} scaffolds)")


# ================================================================
# Cell 6 — Build Split Manifest
# ================================================================
# Create the split_manifest.json mapping each compound_id to
# its fold assignment.

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

# Map compound_id → fold_id
for _, row in df_labeled.iterrows():
    manifest["fold_assignments"][row["compound_id"]] = int(row["fold_id"])

# Save manifest
with open(OUTPUT_MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n✅ Split manifest saved: {OUTPUT_MANIFEST}")
print(f"   Contains {len(manifest['fold_assignments'])} compound → fold mappings")


# ================================================================
# Cell 7 — Merge Fold IDs Back into Full Dataset
# ================================================================
# Add fold_id column to the complete curated dataset (labeled +
# unlabeled). Unlabeled molecules get fold_id = -1.

df["fold_id"] = -1  # default for unlabeled / non-pass molecules

# Merge fold assignments from labeled data
fold_map = df_labeled[["compound_id", "fold_id"]].set_index("compound_id")["fold_id"]
df.loc[df["compound_id"].isin(fold_map.index), "fold_id"] = (
    df.loc[df["compound_id"].isin(fold_map.index), "compound_id"].map(fold_map).values
)

# Save the enriched parquet
df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
print(f"\n✅ Enriched parquet saved: {OUTPUT_PARQUET}")
print(f"   Total molecules: {len(df)}")
print(f"   With fold assignment: {(df['fold_id'] >= 0).sum()}")
print(f"   Without (unlabeled):  {(df['fold_id'] < 0).sum()}")


# ================================================================
# Cell 8 — Save Split Statistics
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

print(f"✅ Split statistics saved: {SPLIT_STATS_FILE}")


# ================================================================
# Cell 9 — Final Verification Summary
# ================================================================

print("\n" + "=" * 60)
print("  PHASE 2 COMPLETE — Scaffold-Aware Splitting")
print("=" * 60)
print(f"\n  Artifacts produced:")
print(f"    1. {OUTPUT_MANIFEST}")
print(f"    2. {OUTPUT_PARQUET}")
print(f"    3. {SPLIT_STATS_FILE}")
print(f"\n  Key results:")
print(f"    • {N_FOLDS}-fold scaffold-aware CV split")
print(f"    • {len(df_labeled)} labeled molecules assigned to folds")
print(f"    • {df_labeled['scaffold_smiles'].nunique()} unique scaffolds")
print(f"    • Zero leakage verified: {'✅ YES' if all_clean else '❌ NO'}")
print(f"\n  Ready for Phase 3 (Feature Engineering) →")
print("=" * 60)
