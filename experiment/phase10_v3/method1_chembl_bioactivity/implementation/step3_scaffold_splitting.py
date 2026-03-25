# ================================================================
# STEP 3 — SCAFFOLD-AWARE SPLITTING (Insecticidal Pipeline)
# ================================================================
# Assign compounds to 5 folds using a greedy balanced algorithm
# based on their Bemis-Murcko scaffolds. Verify 0 leakage.
# Identical logic to Phase 2 (repellency pipeline).
#
# Input:  implementation/artifacts/insecticide_meta.parquet
# Output: implementation/artifacts/split_manifest.json
#         implementation/artifacts/insecticide_meta.parquet (updated)
# ================================================================


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
IMPL_DIR     = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation"
ARTIFACT_DIR = DATA_DIR / "phase10_artifacts"

INPUT_META = ARTIFACT_DIR / "insecticide_meta.parquet"
assert INPUT_META.exists(), f"❌ {INPUT_META} — Run step2 first!"

# Phase 2 split parameters
N_FOLDS = 5
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 3 — SCAFFOLD-AWARE SPLITTING")
print("=" * 65)


# ================================================================
# Cell 2 — Load Metadata
# ================================================================
df_meta = pd.read_parquet(INPUT_META)
scaffolds = df_meta["scaffold_smiles"].values
y = df_meta["insecticidal_active"].values.astype(int)

print(f"\n✅ Loaded metadata for {len(df_meta)} compounds.")
print(f"   Missing scaffolds: {sum(1 for s in scaffolds if pd.isna(s) or s is None)}")


# ================================================================
# Cell 3 — Scaffold Split Algorithm (Phase 2 exact)
# ================================================================
print("\n" + "=" * 65)
print("  SPLIT ASSIGNMENT")
print("=" * 65)

def scaffold_split_kfold(scaffolds, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    n = len(scaffolds)
    fold_ids = np.zeros(n, dtype=int)
    
    # Group indices by scaffold
    scaffold_groups = defaultdict(list)
    no_scaffold = []
    
    for i, scaf in enumerate(scaffolds):
        if scaf is None or pd.isna(scaf) or scaf == "":
            no_scaffold.append(i)
        else:
            scaffold_groups[scaf].append(i)
            
    # Sort groups by size (largest first) to ensure balanced allocation
    groups_by_size = defaultdict(list)
    for scaf, indices in scaffold_groups.items():
        groups_by_size[len(indices)].append((scaf, indices))
        
    sorted_groups = []
    for size in sorted(groups_by_size.keys(), reverse=True):
        items = groups_by_size[size]
        rng.shuffle(items)
        sorted_groups.extend(items)
        
    # Greedy allocation to folds
    fold_sizes = np.zeros(n_folds, dtype=int)
    for scaf, indices in sorted_groups:
        target = int(np.argmin(fold_sizes))
        for idx in indices:
            fold_ids[idx] = target
        fold_sizes[target] += len(indices)
        
    # Handle compounds without scaffolds
    rng.shuffle(no_scaffold)
    for idx in no_scaffold:
        target = int(np.argmin(fold_sizes))
        fold_ids[idx] = target
        fold_sizes[target] += 1
        
    return fold_ids

fold_ids = scaffold_split_kfold(scaffolds, N_FOLDS, RANDOM_SEED)

# Assign to df
df_meta["fold_id"] = fold_ids


# ================================================================
# Cell 4 — Verification and Leakage Check
# ================================================================
print(f"\n── Distribution ──")
for fold in range(N_FOLDS):
    m = (fold_ids == fold)
    n_fold = m.sum()
    n_act  = y[m].sum()
    n_inact = n_fold - n_act
    print(f"  Fold {fold}: {n_fold:>4d} cmpds ({n_act:>3d} active, {n_inact:>3d} inactive)  "
          f"Ratio: {n_act/n_fold:.3f}")

print(f"\n── Leakage Check ──")
all_clean = True
for fold in range(N_FOLDS):
    val_m = (fold_ids == fold)
    trn_m = (fold_ids != fold)
    
    val_scafs = {s for i, s in enumerate(scaffolds[val_m]) if s and not pd.isna(s)}
    trn_scafs = {s for i, s in enumerate(scaffolds[trn_m]) if s and not pd.isna(s)}
    
    intersection = val_scafs.intersection(trn_scafs)
    
    if len(intersection) > 0:
        print(f"  ❌ Fold {fold}: found {len(intersection)} leaking scaffolds!")
        all_clean = False
    else:
        print(f"  ✅ Fold {fold}: 0 leakage.")

if all_clean:
    print(f"\n  🎉 SUCCESS: Zero scaffold leakage detected across all 5 folds.")
else:
    raise ValueError("Scaffold leakage discovered! Check algorithm.")


# ================================================================
# Cell 5 — Save Manifest
# ================================================================
manifest = {
    "num_folds": N_FOLDS,
    "seed": RANDOM_SEED,
    "total_compounds": len(y),
    "fold_distribution": {
        f"fold_{i}": {
            "total": int((fold_ids == i).sum()),
            "active": int(y[fold_ids == i].sum())
        }
        for i in range(N_FOLDS)
    }
}

manifest_path = ARTIFACT_DIR / "split_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=4)
    
# Update metadata
df_meta.to_parquet(INPUT_META, index=False, engine="pyarrow")

print(f"\n✅ Saved: {manifest_path}")
print(f"✅ Updated: {INPUT_META} (added fold_id)")
print(f"\n  → Next: step4_classical_models.py OR step5_ft_transformer.py")
print("=" * 65)
