# *PHASE 3 — FEATURE ENGINEERING*
# 
# Compute classical cheminformatics features for all curated
# molecules: Morgan fingerprints (ECFP6), MACCS keys, and all RDKit 2D descriptors.
# 
# # Input:
# 1. G:/research/ECOAI/experiment/P2_scaffold_splitting/artifacts/curated_molecules_with_splits.parquet
# 
# # Output:
# 1. G:/research/ECOAI/experiment/P3_feature_engineering/artifacts/features_morgan_fp.parquet
# 2. G:/research/ECOAI/experiment/P3_feature_engineering/artifacts/features_maccs_keys.parquet
# 3. G:/research/ECOAI/experiment/P3_feature_engineering/artifacts/features_rdkit_2d.parquet
# 4. G:/research/ECOAI/experiment/P3_feature_engineering/artifacts/features_combined.parquet
# 5. G:/research/ECOAI/experiment/P3_feature_engineering/artifacts/feature_engineering_log.json
# 6. G:/research/ECOAI/experiment/P3_feature_engineering/plots/P3_descriptor_distributions.png
# 7. G:/research/ECOAI/experiment/P3_feature_engineering/plots/P3_descriptor_distributions.jpg


# ================================================================
# S1 — Install Dependencies
# ================================================================
# Uncomment the line below and run ONCE to install all packages.
# !pip install rdkit-pypi pandas pyarrow numpy tqdm scikit-learn matplotlib seaborn


# ================================================================
# S2 — Imports and Configuration
# ================================================================
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(r"G:\research\ECOAI")
P2_ARTIFACTS = PROJECT_ROOT / "experiment" / "P2_scaffold_splitting" / "artifacts"
P3_DIR       = PROJECT_ROOT / "experiment" / "P3_feature_engineering"
ARTIFACTS_DIR = P3_DIR / "artifacts"
PLOTS_DIR     = P3_DIR / "plots"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input / Output
INPUT_PARQUET   = P2_ARTIFACTS / "curated_molecules_with_splits.parquet"
OUTPUT_MORGAN   = ARTIFACTS_DIR / "features_morgan_fp.parquet"
OUTPUT_MACCS    = ARTIFACTS_DIR / "features_maccs_keys.parquet"
OUTPUT_RDKIT2D  = ARTIFACTS_DIR / "features_rdkit_2d.parquet"
OUTPUT_COMBINED = ARTIFACTS_DIR / "features_combined.parquet"
FEATURE_LOG     = ARTIFACTS_DIR / "feature_engineering_log.json"

MORGAN_RADIUS = 3  # ECFP6
MORGAN_NBITS  = 2048

assert INPUT_PARQUET.exists(), f"[ERROR] Not found: {INPUT_PARQUET}"

print("=" * 60)
print("  P3 — ADVANCED FEATURE ENGINEERING")
print("=" * 60)
print(f"  Input:        {INPUT_PARQUET.name}")
print(f"  Morgan FP:    radius={MORGAN_RADIUS}, bits={MORGAN_NBITS}")
print(f"  MACCS Keys:   166 bits")
print(f"  RDKit 2D:     All ~200 available descriptors")


# ================================================================
# S3 — Load Curated Data
# ================================================================
df = pd.read_parquet(INPUT_PARQUET)

df_pass = df[df["qc_status"] == "pass"].copy().reset_index(drop=True)

print(f"\n[OK] Loaded {len(df)} molecules total, {len(df_pass)} QC-pass.")
print(f"   Labeled:   {df_pass['repellent_active'].notna().sum()}")
print(f"   Unlabeled: {df_pass['repellent_active'].isna().sum()}")


# ================================================================
# S4 — Convert SMILES to RDKit Mol Objects
# ================================================================
print("\n--- Converting SMILES to Mol objects ---")
mols = []
valid_mask = []

for smiles in tqdm(df_pass["canonical_smiles"], desc="  Parsing"):
    if pd.isna(smiles):
        mols.append(None)
        valid_mask.append(False)
        continue
    mol = Chem.MolFromSmiles(smiles)
    mols.append(mol)
    valid_mask.append(mol is not None)

df_pass["_mol"] = mols
df_pass["_valid"] = valid_mask

n_valid = sum(valid_mask)
n_invalid = len(valid_mask) - n_valid
print(f"  [OK] Valid: {n_valid},  Failed: {n_invalid}")


# ================================================================
# S5 — Compute Morgan Fingerprints (ECFP6)
# ================================================================
print(f"\n--- Computing Morgan Fingerprints (r={MORGAN_RADIUS}, {MORGAN_NBITS} bits) ---")

fp_array = np.zeros((len(df_pass), MORGAN_NBITS), dtype=np.uint8)

for idx, mol in enumerate(tqdm(df_pass["_mol"], desc="  Morgan FP")):
    if mol is None:
        continue
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS
        )
        arr = np.zeros(MORGAN_NBITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array[idx] = arr
    except Exception as e:
         print(f"  [WARNING] FP failed for {df_pass.iloc[idx]['compound_id']}: {e}")

fp_columns = [f"mfp_{i}" for i in range(MORGAN_NBITS)]
df_morgan = pd.DataFrame(fp_array, columns=fp_columns)
df_morgan.insert(0, "compound_id", df_pass["compound_id"].values)

bits_on = fp_array.sum(axis=1)
print(f"  [OK] Morgan fingerprints computed.")
print(f"     Bits ON per molecule: mean={bits_on.mean():.1f}, "
      f"min={bits_on.min()}, max={bits_on.max()}")


# ================================================================
# S6 — Compute MACCS Keys
# ================================================================
print(f"\n--- Computing MACCS Keys (166 bits) ---")

maccs_array = np.zeros((len(df_pass), 167), dtype=np.uint8)

for idx, mol in enumerate(tqdm(df_pass["_mol"], desc="  MACCS Keys")):
    if mol is None:
        continue
    try:
        maccs = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        arr = np.zeros(167, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(maccs, arr)
        maccs_array[idx] = arr
    except Exception as e:
         print(f"  [WARNING] MACCS failed for {df_pass.iloc[idx]['compound_id']}: {e}")

# MACCS keys are 1-indexed in RDKit (0 bit is ignored)
maccs_array = maccs_array[:, 1:]
maccs_columns = [f"maccs_{i}" for i in range(1, 167)]
df_maccs = pd.DataFrame(maccs_array, columns=maccs_columns)
df_maccs.insert(0, "compound_id", df_pass["compound_id"].values)

print(f"  [OK] MACCS keys computed.")


# ================================================================
# S7 — Compute All RDKit 2D Descriptors
# ================================================================
RDKIT_2D_DESCRIPTORS = Descriptors.descList

print(f"\n--- Computing {len(RDKIT_2D_DESCRIPTORS)} RDKit 2D Descriptors ---")

desc_data = {name: [] for name, _ in RDKIT_2D_DESCRIPTORS}

for mol in tqdm(df_pass["_mol"], desc="  RDKit 2D"):
    for name, func in RDKIT_2D_DESCRIPTORS:
        if mol is None:
            desc_data[name].append(np.nan)
        else:
            try:
                val = func(mol)
                desc_data[name].append(val)
            except Exception:
                desc_data[name].append(np.nan)

df_rdkit2d = pd.DataFrame(desc_data)
df_rdkit2d.insert(0, "compound_id", df_pass["compound_id"].values)

null_counts = df_rdkit2d.isnull().sum()
problematic = null_counts[null_counts > 0]
if len(problematic) > 0:
    print(f"\n  [WARNING] Descriptors with NaN values (will be imputed later):")
    for col, cnt in problematic.items():
        if col != "compound_id":
            print(f"     {col}: {cnt} NaN")

df_rdkit2d = df_rdkit2d.replace([np.inf, -np.inf], np.nan)
print(f"  [OK] RDKit 2D descriptors computed: {len(RDKIT_2D_DESCRIPTORS)} features")


# ================================================================
# S8 — Build Combined Feature Matrix
# ================================================================
df_combined = df_morgan.merge(df_maccs, on="compound_id", how="inner").merge(df_rdkit2d, on="compound_id", how="inner")

meta_cols = [
    "compound_id", "canonical_smiles", "source_dataset",
    "activity_class", "repellent_active", "scaffold_smiles", "fold_id", "qc_status",
]
meta_available = [c for c in meta_cols if c in df_pass.columns]
df_meta = df_pass[meta_available].copy()

df_combined = df_meta.merge(df_combined, on="compound_id", how="inner")
n_features = len(df_combined.columns) - len(meta_available)

print(f"\n[OK] Combined feature matrix: {len(df_combined)} molecules × {n_features} features")


# ================================================================
# S9 — Feature Correlation Check (RDKit 2D only)
# ================================================================
print("\n--- Checking for highly correlated descriptor pairs ---")

desc_numeric = df_rdkit2d.drop(columns=["compound_id"]).select_dtypes(include=[np.number])
corr_matrix = desc_numeric.corr().abs()

high_corr_pairs = []
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if corr_matrix.iloc[i, j] > 0.98:
            high_corr_pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print(f"  [WARNING] {len(high_corr_pairs)} descriptor pairs with |r| > 0.98 (showing top 15):")
    for d1, d2, r in sorted(high_corr_pairs, key=lambda x: -x[2])[:15]:
        print(f"     {d1:25s} ↔ {d2:25s}  r={r:.3f}")
else:
    print("  [OK] No highly correlated pairs found.")


# ================================================================
# S10 — Save Feature Files
# ================================================================
df_morgan.to_parquet(OUTPUT_MORGAN, index=False, engine="pyarrow")
print(f"\n[OK] Saved Morgan FP:    {OUTPUT_MORGAN.name}  ({OUTPUT_MORGAN.stat().st_size / 1024:.0f} KB)")

df_maccs.to_parquet(OUTPUT_MACCS, index=False, engine="pyarrow")
print(f"[OK] Saved MACCS Keys:   {OUTPUT_MACCS.name}  ({OUTPUT_MACCS.stat().st_size / 1024:.0f} KB)")

df_rdkit2d.to_parquet(OUTPUT_RDKIT2D, index=False, engine="pyarrow")
print(f"[OK] Saved RDKit 2D:     {OUTPUT_RDKIT2D.name}  ({OUTPUT_RDKIT2D.stat().st_size / 1024:.0f} KB)")

df_combined.to_parquet(OUTPUT_COMBINED, index=False, engine="pyarrow")
print(f"[OK] Saved combined:     {OUTPUT_COMBINED.name}  ({OUTPUT_COMBINED.stat().st_size / 1024 / 1024:.1f} MB)")


# ================================================================
# S11 — Generate Descriptor Distribution Plots (PNG + JPG)
# ================================================================
# Plot molecular weight (MolWt) and LogP distribution for Active vs Decoy
df_labeled_pass = df_combined[df_combined["repellent_active"].notna()]
active_wt = df_labeled_pass[df_labeled_pass["repellent_active"] == 1]["MolWt"].dropna()
decoy_wt = df_labeled_pass[df_labeled_pass["repellent_active"] == 0]["MolWt"].dropna()

active_logp = df_labeled_pass[df_labeled_pass["repellent_active"] == 1]["MolLogP"].dropna()
decoy_logp = df_labeled_pass[df_labeled_pass["repellent_active"] == 0]["MolLogP"].dropna()

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 12
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Molecular Weight distribution boxplot
axes[0].boxplot([active_wt, decoy_wt], labels=["Active Repellents (1)", "Inactives (0)"], patch_artist=True,
                boxprops=dict(facecolor="#17a589", alpha=0.7), medianprops=dict(color="black", linewidth=2))
axes[0].set_title("Molecular Weight Distribution", fontsize=13, fontweight="bold", color="black")
axes[0].set_ylabel("MolWt (g/mol)", fontsize=12, fontweight="bold", color="black")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# MolLogP distribution boxplot
axes[1].boxplot([active_logp, decoy_logp], labels=["Active Repellents (1)", "Inactives (0)"], patch_artist=True,
                boxprops=dict(facecolor="#1b4f72", alpha=0.7), medianprops=dict(color="black", linewidth=2))
axes[1].set_title("Partition Coefficient (MolLogP) Distribution", fontsize=13, fontweight="bold", color="black")
axes[1].set_ylabel("MolLogP", fontsize=12, fontweight="bold", color="black")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

for ax in axes:
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
        tick.set_color("black")
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_color("black")
        tick.set_fontsize(12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

fig.suptitle("Key Cheminformatics Properties: Active vs Inactive Molecules", fontsize=15, fontweight="bold", color="black")
plt.tight_layout(pad=2.0)

plot_path_png = PLOTS_DIR / "P3_descriptor_distributions.png"
plot_path_jpg = PLOTS_DIR / "P3_descriptor_distributions.jpg"

plt.savefig(plot_path_png, dpi=300)
plt.savefig(plot_path_jpg, dpi=300)
plt.close()

print(f"\n[OK] Saved plots:\n  - {plot_path_png}\n  - {plot_path_jpg}")


# ================================================================
# S12 — Feature Engineering Log
# ================================================================
feature_log = {
    "pipeline": "P3 — Advanced Feature Engineering",
    "morgan_config": {
        "radius": MORGAN_RADIUS,
        "n_bits": MORGAN_NBITS,
        "description": f"ECFP{2 * MORGAN_RADIUS} fingerprint",
    },
    "maccs_keys": {
        "bits": 166
    },
    "rdkit_2d_descriptors": {
        "count": len(RDKIT_2D_DESCRIPTORS),
        "names": [name for name, _ in RDKIT_2D_DESCRIPTORS],
    },
    "total_features": n_features,
    "molecules_processed": len(df_pass),
    "valid_molecules": n_valid,
    "high_corr_pairs_above_0.98": len(high_corr_pairs),
    "output_files": {
        "morgan_fp": str(OUTPUT_MORGAN),
        "maccs_keys": str(OUTPUT_MACCS),
        "rdkit_2d": str(OUTPUT_RDKIT2D),
        "combined": str(OUTPUT_COMBINED),
    },
}

with open(FEATURE_LOG, "w") as f:
    json.dump(feature_log, f, indent=2)

print(f"[OK] Feature log saved: {FEATURE_LOG}")


# ================================================================
# S13 — Final Summary
# ================================================================
print("\n" + "=" * 60)
print("  P3 COMPLETE — Advanced Feature Engineering")
print("=" * 60)
print(f"  Artifact Directory: {ARTIFACTS_DIR}")
print(f"  Plots Directory:    {PLOTS_DIR}")
print()
print(f"  Artifacts Generated:")
print(f"    1. {OUTPUT_MORGAN.name} (Morgan fingerprints - {MORGAN_NBITS} bits)")
print(f"    2. {OUTPUT_MACCS.name} (MACCS Keys - 166 bits)")
print(f"    3. {OUTPUT_RDKIT2D.name} (RDKit 2D descriptors - {len(RDKIT_2D_DESCRIPTORS)} features)")
print(f"    4. {OUTPUT_COMBINED.name} (Combined metadata + {n_features} feature matrix)")
print(f"    5. {FEATURE_LOG.name} (Feature engineering logs)")
print(f"    6. P3_descriptor_distributions.png / .jpg (Descriptor boxplots)")
print()
print(f"  Summary:")
print(f"    - Total processed:      {len(df_pass)}")
print(f"    - Total features:       {n_features}")
print(f"    - Collinear pairs (>0.98): {len(high_corr_pairs)}")
print("=" * 60)
