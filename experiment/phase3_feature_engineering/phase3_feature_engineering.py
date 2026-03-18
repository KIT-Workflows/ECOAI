# ================================================================
# PHASE 3 — FEATURE ENGINEERING
# ================================================================
# Compute classical cheminformatics features for all curated
# molecules: Morgan fingerprints (ECFP4) and RDKit 2D descriptors.
#
# Input:  Datasets/data/curated_molecules_with_splits.parquet
# Output: Datasets/data/features_morgan_fp.parquet
#         Datasets/data/features_rdkit_2d.parquet
#         Datasets/data/features_combined.parquet
#
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# !pip install rdkit-pypi pandas pyarrow numpy tqdm scikit-learn


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import DataStructs
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
PHASE3_DIR   = PROJECT_ROOT / "experiment" / "phase3_feature_engineering"

# ── Input ──────────────────────────────────────────────────
INPUT_PARQUET = DATA_DIR / "curated_molecules_with_splits.parquet"

# ── Outputs ────────────────────────────────────────────────
OUTPUT_MORGAN   = DATA_DIR / "features_morgan_fp.parquet"
OUTPUT_RDKIT2D  = DATA_DIR / "features_rdkit_2d.parquet"
OUTPUT_COMBINED = DATA_DIR / "features_combined.parquet"
FEATURE_LOG     = PHASE3_DIR / "feature_engineering_log.json"

# ── Morgan FP Configuration ───────────────────────────────
MORGAN_RADIUS = 2
MORGAN_NBITS  = 2048

# ── Verify ─────────────────────────────────────────────────
assert INPUT_PARQUET.exists(), f"❌ Not found: {INPUT_PARQUET}\n   Run Phase 2 first!"
PHASE3_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 3 — FEATURE ENGINEERING")
print("=" * 60)
print(f"  Input:        {INPUT_PARQUET.name}")
print(f"  Morgan FP:    radius={MORGAN_RADIUS}, bits={MORGAN_NBITS}")


# ================================================================
# Cell 2 — Load Curated Data
# ================================================================

df = pd.read_parquet(INPUT_PARQUET)

# Filter to QC-pass molecules only
df_pass = df[df["qc_status"] == "pass"].copy().reset_index(drop=True)

print(f"\n✅ Loaded {len(df)} molecules total, {len(df_pass)} QC-pass.")
print(f"   Labeled:   {df_pass['repellent_active'].notna().sum()}")
print(f"   Unlabeled: {df_pass['repellent_active'].isna().sum()}")


# ================================================================
# Cell 3 — Convert SMILES to RDKit Mol Objects
# ================================================================
# We need Mol objects for fingerprint and descriptor computation.
# Cache them here to avoid repeated parsing.

print("\n── Converting SMILES to Mol objects ──")
mols = []
valid_mask = []

for idx, smiles in enumerate(tqdm(df_pass["canonical_smiles"], desc="  Parsing")):
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
print(f"  ✅ Valid: {n_valid},  Failed: {n_invalid}")

if n_invalid > 0:
    failed = df_pass[~df_pass["_valid"]][["compound_id", "canonical_smiles"]]
    print(f"  ⚠️  Failed molecules:")
    print(failed.to_string(index=False))


# ================================================================
# Cell 4 — Compute Morgan Fingerprints (ECFP4)
# ================================================================
# Morgan fingerprints with radius=2, 2048 bits ≈ ECFP4.
# These capture circular substructures around each atom.

print(f"\n── Computing Morgan Fingerprints (r={MORGAN_RADIUS}, {MORGAN_NBITS} bits) ──")

fp_array = np.zeros((len(df_pass), MORGAN_NBITS), dtype=np.uint8)

for idx, mol in enumerate(tqdm(df_pass["_mol"], desc="  Morgan FP")):
    if mol is None:
        continue  # row stays all zeros
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS
        )
        arr = np.zeros(MORGAN_NBITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array[idx] = arr
    except Exception as e:
        print(f"  ⚠️  FP failed for {df_pass.iloc[idx]['compound_id']}: {e}")

# Create DataFrame with descriptive column names
fp_columns = [f"mfp_{i}" for i in range(MORGAN_NBITS)]
df_morgan = pd.DataFrame(fp_array, columns=fp_columns)
df_morgan.insert(0, "compound_id", df_pass["compound_id"].values)

# ── Statistics ──
bits_on = fp_array.sum(axis=1)
print(f"\n  ✅ Morgan fingerprints computed.")
print(f"     Bits ON per molecule: mean={bits_on.mean():.1f}, "
      f"min={bits_on.min()}, max={bits_on.max()}")


# ================================================================
# Cell 5 — Compute RDKit 2D Descriptors
# ================================================================
# A curated set of 2D molecular descriptors from RDKit.

# Define the descriptor set
RDKIT_2D_DESCRIPTORS = [
    ("MolWt",            Descriptors.MolWt),
    ("ExactMolWt",       Descriptors.ExactMolWt),
    ("HeavyAtomCount",   Descriptors.HeavyAtomCount),
    ("NumHAcceptors",    Descriptors.NumHAcceptors),
    ("NumHDonors",       Descriptors.NumHDonors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("RingCount",        Descriptors.RingCount),
    ("TPSA",             Descriptors.TPSA),
    ("MolLogP",          Descriptors.MolLogP),
    ("MolMR",            Descriptors.MolMR),
    ("FractionCSP3",     Descriptors.FractionCSP3),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
    ("MaxPartialCharge",  Descriptors.MaxPartialCharge),
    ("MinPartialCharge",  Descriptors.MinPartialCharge),
    ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge),
    ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge),
    ("BalabanJ",         Descriptors.BalabanJ),
    ("BertzCT",          Descriptors.BertzCT),
    ("Chi0",             Descriptors.Chi0),
    ("Chi0n",            Descriptors.Chi0n),
    ("Chi0v",            Descriptors.Chi0v),
    ("Chi1",             Descriptors.Chi1),
    ("Chi1n",            Descriptors.Chi1n),
    ("Chi1v",            Descriptors.Chi1v),
    ("HallKierAlpha",    Descriptors.HallKierAlpha),
    ("Kappa1",           Descriptors.Kappa1),
    ("Kappa2",           Descriptors.Kappa2),
    ("Kappa3",           Descriptors.Kappa3),
    ("LabuteASA",        Descriptors.LabuteASA),
    ("PEOE_VSA1",        Descriptors.PEOE_VSA1),
    ("PEOE_VSA2",        Descriptors.PEOE_VSA2),
    ("PEOE_VSA3",        Descriptors.PEOE_VSA3),
    ("PEOE_VSA6",        Descriptors.PEOE_VSA6),
    ("PEOE_VSA7",        Descriptors.PEOE_VSA7),
    ("PEOE_VSA8",        Descriptors.PEOE_VSA8),
    ("SMR_VSA1",         Descriptors.SMR_VSA1),
    ("SMR_VSA5",         Descriptors.SMR_VSA5),
    ("SMR_VSA7",         Descriptors.SMR_VSA7),
    ("SlogP_VSA2",       Descriptors.SlogP_VSA2),
    ("SlogP_VSA3",       Descriptors.SlogP_VSA3),
    ("SlogP_VSA5",       Descriptors.SlogP_VSA5),
    ("EState_VSA1",      Descriptors.EState_VSA1),
    ("EState_VSA2",      Descriptors.EState_VSA2),
    ("VSA_EState1",      Descriptors.VSA_EState1),
    ("VSA_EState2",      Descriptors.VSA_EState2),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("NumAromaticHeterocycles", Descriptors.NumAromaticHeterocycles),
    ("NumSaturatedHeterocycles", Descriptors.NumSaturatedHeterocycles),
    ("NHOHCount",        Descriptors.NHOHCount),
    ("NOCount",          Descriptors.NOCount),
    ("NumHeteroatoms",   Descriptors.NumHeteroatoms),
]

print(f"\n── Computing {len(RDKIT_2D_DESCRIPTORS)} RDKit 2D Descriptors ──")

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

# ── Check for problematic descriptors ──────────────────────
null_counts = df_rdkit2d.isnull().sum()
inf_mask = df_rdkit2d.select_dtypes(include=[np.number]).apply(
    lambda x: np.isinf(x).sum()
)

problematic = null_counts[null_counts > 0]
if len(problematic) > 0:
    print(f"\n  ⚠️  Descriptors with NaN values:")
    for col, cnt in problematic.items():
        if col != "compound_id":
            print(f"     {col}: {cnt} NaN")

inf_cols = inf_mask[inf_mask > 0]
if len(inf_cols) > 0:
    print(f"\n  ⚠️  Descriptors with Inf values:")
    for col, cnt in inf_cols.items():
        print(f"     {col}: {cnt} Inf")
    # Replace Inf with NaN for clean downstream use
    df_rdkit2d = df_rdkit2d.replace([np.inf, -np.inf], np.nan)
    print("     → Inf values replaced with NaN")

print(f"\n  ✅ RDKit 2D descriptors computed: {len(RDKIT_2D_DESCRIPTORS)} features")


# ================================================================
# Cell 6 — Descriptor Summary Statistics
# ================================================================

print("\n" + "=" * 60)
print("  DESCRIPTOR SUMMARY (first 20)")
print("=" * 60)

desc_cols = [c for c in df_rdkit2d.columns if c != "compound_id"]
summary = df_rdkit2d[desc_cols[:20]].describe().T
summary["null_pct"] = df_rdkit2d[desc_cols[:20]].isnull().mean() * 100
print(summary[["mean", "std", "min", "max", "null_pct"]].round(2).to_string())


# ================================================================
# Cell 7 — Build Combined Feature Matrix
# ================================================================
# Merge Morgan FP + RDKit 2D descriptors into a single feature
# table for modeling.

df_combined = df_morgan.merge(df_rdkit2d, on="compound_id", how="inner")

# Also merge in the metadata we need for modeling
meta_cols = [
    "compound_id", "canonical_smiles", "source_dataset",
    "repellent_active", "scaffold_smiles", "fold_id", "qc_status",
]
meta_available = [c for c in meta_cols if c in df_pass.columns]
df_meta = df_pass[meta_available].copy()

df_combined = df_meta.merge(df_combined, on="compound_id", how="inner")

n_features = len(df_combined.columns) - len(meta_available)
print(f"\n✅ Combined feature matrix: {len(df_combined)} molecules × {n_features} features")
print(f"   Morgan FP bits: {MORGAN_NBITS}")
print(f"   RDKit 2D descriptors: {len(RDKIT_2D_DESCRIPTORS)}")
print(f"   Total features: {n_features}")


# ================================================================
# Cell 8 — Feature Correlation Check (RDKit 2D only)
# ================================================================
# Check for highly correlated descriptor pairs that might cause
# redundancy.

print("\n── Checking for highly correlated descriptor pairs ──")

desc_numeric = df_rdkit2d.drop(columns=["compound_id"]).select_dtypes(include=[np.number])
corr_matrix = desc_numeric.corr().abs()

# Find pairs with |r| > 0.95
high_corr_pairs = []
cols = corr_matrix.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print(f"  ⚠️  {len(high_corr_pairs)} descriptor pairs with |r| > 0.95:")
    for d1, d2, r in sorted(high_corr_pairs, key=lambda x: -x[2])[:15]:
        print(f"     {d1:25s} ↔ {d2:25s}  r={r:.3f}")
    if len(high_corr_pairs) > 15:
        print(f"     ... and {len(high_corr_pairs) - 15} more")
    print("  ℹ️  These are kept for now; models can handle collinearity.")
else:
    print("  ✅ No highly correlated pairs found.")


# ================================================================
# Cell 9 — Save Feature Files
# ================================================================

# Save Morgan fingerprints
df_morgan.to_parquet(OUTPUT_MORGAN, index=False, engine="pyarrow")
print(f"\n✅ Saved Morgan FP:    {OUTPUT_MORGAN.name}  ({OUTPUT_MORGAN.stat().st_size / 1024:.0f} KB)")

# Save RDKit 2D descriptors
df_rdkit2d.to_parquet(OUTPUT_RDKIT2D, index=False, engine="pyarrow")
print(f"✅ Saved RDKit 2D:     {OUTPUT_RDKIT2D.name}  ({OUTPUT_RDKIT2D.stat().st_size / 1024:.0f} KB)")

# Save combined features
df_combined.to_parquet(OUTPUT_COMBINED, index=False, engine="pyarrow")
print(f"✅ Saved combined:     {OUTPUT_COMBINED.name}  ({OUTPUT_COMBINED.stat().st_size / 1024:.0f} KB)")


# ================================================================
# Cell 10 — Feature Engineering Log
# ================================================================

feature_log = {
    "pipeline": "Phase 3 — Feature Engineering",
    "morgan_config": {
        "radius": MORGAN_RADIUS,
        "n_bits": MORGAN_NBITS,
        "description": f"ECFP{2 * MORGAN_RADIUS} fingerprint",
    },
    "rdkit_2d_descriptors": {
        "count": len(RDKIT_2D_DESCRIPTORS),
        "names": [name for name, _ in RDKIT_2D_DESCRIPTORS],
    },
    "total_features": n_features,
    "molecules_processed": len(df_pass),
    "valid_molecules": n_valid,
    "high_corr_pairs_above_0.95": len(high_corr_pairs),
    "output_files": {
        "morgan_fp": str(OUTPUT_MORGAN),
        "rdkit_2d": str(OUTPUT_RDKIT2D),
        "combined": str(OUTPUT_COMBINED),
    },
}

with open(FEATURE_LOG, "w") as f:
    json.dump(feature_log, f, indent=2)

print(f"✅ Feature log saved: {FEATURE_LOG}")


# ================================================================
# Cell 11 — Final Summary
# ================================================================

print("\n" + "=" * 60)
print("  PHASE 3 COMPLETE — Feature Engineering")
print("=" * 60)
print(f"\n  Artifacts produced:")
print(f"    1. {OUTPUT_MORGAN.name}   — Morgan fingerprints ({MORGAN_NBITS} bits)")
print(f"    2. {OUTPUT_RDKIT2D.name}  — RDKit 2D descriptors ({len(RDKIT_2D_DESCRIPTORS)} features)")
print(f"    3. {OUTPUT_COMBINED.name} — Combined feature matrix ({n_features} features)")
print(f"    4. {FEATURE_LOG.name}")
print(f"\n  Ready for Phase 4 (Model Training) →")
print("=" * 60)
