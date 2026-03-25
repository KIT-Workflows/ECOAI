# ================================================================
# STEP 1 — DATA LOADING (Insecticidal Pipeline)
# ================================================================
# Load insecticide labels from Phase 10 Prep and verify integrity.
#
# Input:  Datasets/data/insecticide_labels.parquet
# Output: implementation/artifacts/insecticide_data.parquet  (verified)
# ================================================================


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
IMPL_DIR     = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation"
ARTIFACT_DIR = DATA_DIR / "phase10_artifacts"

INPUT_LABELS = DATA_DIR / "insecticide_labels.parquet"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
assert INPUT_LABELS.exists(), f"❌ {INPUT_LABELS} — Run data_prep first!"

print("=" * 65)
print("  STEP 1 — DATA LOADING")
print("=" * 65)


# ================================================================
# Cell 2 — Load and Verify Labels
# ================================================================
df = pd.read_parquet(INPUT_LABELS)

print(f"\n✅ Loaded {len(df)} compounds from Phase 10 Prep.")
print(f"   Columns: {list(df.columns)}")
print(f"\n   Label distribution:")
print(f"     active   (1): {(df['insecticidal_active'] == 1).sum()}")
print(f"     inactive (0): {(df['insecticidal_active'] == 0).sum()}")
print(f"     ratio:        {(df['insecticidal_active'] == 1).mean():.3f}")

print(f"\n   Source:")
print(f"     LifeChemicals matched: {df['in_lifechem'].sum()}")
print(f"     External (ChEMBL):     {(~df['in_lifechem']).sum()}")

# Verify required columns
required = ["compound_id", "inchikey", "canonical_smiles", "insecticidal_active", "in_lifechem"]
for col in required:
    assert col in df.columns, f"❌ Missing column: {col}"
print(f"\n   ✅ All required columns present.")


# ================================================================
# Cell 3 — Data Quality Checks
# ================================================================
print("\n" + "=" * 65)
print("  DATA QUALITY CHECKS")
print("=" * 65)

# No duplicates by InChIKey
n_dupes = df["inchikey"].duplicated().sum()
print(f"  Duplicate InChIKeys: {n_dupes}")
if n_dupes > 0:
    print(f"  ⚠️  Removing {n_dupes} duplicates (keeping first)...")
    df = df.drop_duplicates(subset="inchikey", keep="first").reset_index(drop=True)

# No missing SMILES
n_missing = df["canonical_smiles"].isna().sum()
print(f"  Missing SMILES:      {n_missing}")
if n_missing > 0:
    df = df[df["canonical_smiles"].notna()].reset_index(drop=True)

# No missing labels
n_no_label = df["insecticidal_active"].isna().sum()
print(f"  Missing labels:      {n_no_label}")

# Class balance
n_a = (df["insecticidal_active"] == 1).sum()
n_i = (df["insecticidal_active"] == 0).sum()
imbalance = max(n_a, n_i) / min(n_a, n_i) if min(n_a, n_i) > 0 else float("inf")
print(f"\n  Class balance: {n_a} active / {n_i} inactive (ratio {imbalance:.2f}:1)")
if imbalance > 5:
    print(f"  ⚠️  HIGH IMBALANCE — consider class weighting or resampling.")
elif imbalance > 2:
    print(f"  ⚠️  Moderate imbalance — class_weight='balanced' recommended.")
else:
    print(f"  ✅ Reasonably balanced.")

print(f"\n  ✅ Final dataset: {len(df)} compounds")


# ================================================================
# Cell 4 — Save Verified Data
# ================================================================
output_path = ARTIFACT_DIR / "insecticide_data.parquet"
df.to_parquet(output_path, index=False, engine="pyarrow")

print(f"\n✅ Saved: {output_path}")
print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
print(f"\n  → Next: step2_feature_engineering.py")
print("=" * 65)
