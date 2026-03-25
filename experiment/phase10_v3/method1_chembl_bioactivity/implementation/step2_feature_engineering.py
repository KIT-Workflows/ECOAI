# ================================================================
# STEP 2 — FEATURE ENGINEERING (Insecticidal Pipeline)
# ================================================================
# Compute Morgan FP (2048 bits) + 54 RDKit 2D descriptors.
# Identical feature set to Phase 3 (repellency pipeline).
#
# Input:  implementation/artifacts/insecticide_data.parquet
# Output: implementation/artifacts/insecticide_features.npz  (X matrix)
#         implementation/artifacts/insecticide_meta.parquet   (labels + IDs)
# ================================================================


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
IMPL_DIR     = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "implementation"
ARTIFACT_DIR = DATA_DIR / "phase10_artifacts"

INPUT_DATA = ARTIFACT_DIR / "insecticide_data.parquet"
assert INPUT_DATA.exists(), f"❌ {INPUT_DATA} — Run step1 first!"

MORGAN_RADIUS = 2
MORGAN_NBITS  = 2048
RANDOM_SEED   = 42

print("=" * 65)
print("  STEP 2 — FEATURE ENGINEERING")
print("=" * 65)


# ================================================================
# Cell 2 — Load Verified Data
# ================================================================
df = pd.read_parquet(INPUT_DATA)
print(f"\n✅ Loaded {len(df)} compounds.")


# ================================================================
# Cell 3 — Define RDKit 2D Descriptors (Phase 3 exact list)
# ================================================================
RDKIT_2D_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt), ("ExactMolWt", Descriptors.ExactMolWt),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("RingCount", Descriptors.RingCount), ("TPSA", Descriptors.TPSA),
    ("MolLogP", Descriptors.MolLogP), ("MolMR", Descriptors.MolMR),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge),
    ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge),
    ("BalabanJ", Descriptors.BalabanJ), ("BertzCT", Descriptors.BertzCT),
    ("Chi0", Descriptors.Chi0), ("Chi0n", Descriptors.Chi0n),
    ("Chi0v", Descriptors.Chi0v), ("Chi1", Descriptors.Chi1),
    ("Chi1n", Descriptors.Chi1n), ("Chi1v", Descriptors.Chi1v),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("Kappa1", Descriptors.Kappa1), ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3), ("LabuteASA", Descriptors.LabuteASA),
    ("PEOE_VSA1", Descriptors.PEOE_VSA1), ("PEOE_VSA2", Descriptors.PEOE_VSA2),
    ("PEOE_VSA3", Descriptors.PEOE_VSA3), ("PEOE_VSA6", Descriptors.PEOE_VSA6),
    ("PEOE_VSA7", Descriptors.PEOE_VSA7), ("PEOE_VSA8", Descriptors.PEOE_VSA8),
    ("SMR_VSA1", Descriptors.SMR_VSA1), ("SMR_VSA5", Descriptors.SMR_VSA5),
    ("SMR_VSA7", Descriptors.SMR_VSA7), ("SlogP_VSA2", Descriptors.SlogP_VSA2),
    ("SlogP_VSA3", Descriptors.SlogP_VSA3), ("SlogP_VSA5", Descriptors.SlogP_VSA5),
    ("EState_VSA1", Descriptors.EState_VSA1), ("EState_VSA2", Descriptors.EState_VSA2),
    ("VSA_EState1", Descriptors.VSA_EState1), ("VSA_EState2", Descriptors.VSA_EState2),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("NumAromaticHeterocycles", Descriptors.NumAromaticHeterocycles),
    ("NumSaturatedHeterocycles", Descriptors.NumSaturatedHeterocycles),
    ("NHOHCount", Descriptors.NHOHCount), ("NOCount", Descriptors.NOCount),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
]

morgan_cols = [f"mfp_{i}" for i in range(MORGAN_NBITS)]
rdkit_cols  = [name for name, _ in RDKIT_2D_DESCRIPTORS]
feature_cols = morgan_cols + rdkit_cols

print(f"  Feature set: {MORGAN_NBITS} Morgan FP + {len(RDKIT_2D_DESCRIPTORS)} RDKit 2D = {len(feature_cols)} total")


# ================================================================
# Cell 4 — Compute Features + Scaffolds
# ================================================================
# For each molecule: Morgan FP + RDKit 2D + Bemis-Murcko scaffold

print(f"\n── Computing features for {len(df)} molecules ──")

from rdkit.Chem.Scaffolds import MurckoScaffold

n = len(df)
X = np.zeros((n, len(feature_cols)), dtype=np.float64)
scaffolds = []
failed_mols = []

for idx, smi in enumerate(tqdm(df["canonical_smiles"], desc="  Features")):
    if pd.isna(smi) or not smi:
        scaffolds.append(None)
        X[idx, MORGAN_NBITS:] = np.nan
        failed_mols.append(idx)
        continue

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        scaffolds.append(None)
        X[idx, MORGAN_NBITS:] = np.nan
        failed_mols.append(idx)
        continue

    # Morgan fingerprint (Phase 3 pattern)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS)
        arr = np.zeros(MORGAN_NBITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[idx, :MORGAN_NBITS] = arr
    except Exception as e:
        failed_mols.append(idx)

    # RDKit 2D descriptors (Phase 3 pattern: Inf→NaN)
    for j, (name, func) in enumerate(RDKIT_2D_DESCRIPTORS):
        try:
            val = func(mol)
            X[idx, MORGAN_NBITS + j] = np.nan if np.isinf(val) else val
        except Exception:
            X[idx, MORGAN_NBITS + j] = np.nan

    # Bemis-Murcko scaffold (Phase 2 pattern)
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds.append(scaf if scaf else None)
    except Exception:
        scaffolds.append(None)


# ================================================================
# Cell 5 — Feature Statistics
# ================================================================
print(f"\n" + "=" * 65)
print("  FEATURE STATISTICS")
print("=" * 65)

# Morgan FP stats
bits_on = X[:, :MORGAN_NBITS].sum(axis=1)
print(f"  Morgan FP bits ON/molecule: mean={bits_on.mean():.1f}, min={bits_on.min():.0f}, max={bits_on.max():.0f}")

# RDKit descriptor stats
X_rdkit = X[:, MORGAN_NBITS:]
n_nan = np.isnan(X_rdkit).sum(axis=0)
problematic = [(rdkit_cols[j], int(n_nan[j])) for j in range(len(rdkit_cols)) if n_nan[j] > 0]
if problematic:
    print(f"\n  ⚠️  Descriptors with NaN:")
    for col, cnt in problematic:
        print(f"     {col}: {cnt}")
else:
    print(f"  ✅ No NaN in RDKit descriptors.")

print(f"\n  Scaffolds found: {sum(1 for s in scaffolds if s)}/{n}")
print(f"  Failed molecules: {len(set(failed_mols))}")


# ================================================================
# Cell 6 — Save Artifacts
# ================================================================
# Save feature matrix as .npz (compact) and metadata as parquet

# Feature matrix
feat_path = ARTIFACT_DIR / "insecticide_features.npz"
np.savez_compressed(feat_path, X=X)
print(f"\n✅ Features saved: {feat_path}  ({feat_path.stat().st_size/1024:.1f} KB)")

# Metadata + scaffold + feature column names
df_meta = df[["compound_id", "inchikey", "canonical_smiles",
              "insecticidal_active", "in_lifechem"]].copy()
df_meta["scaffold_smiles"] = scaffolds

meta_path = ARTIFACT_DIR / "insecticide_meta.parquet"
df_meta.to_parquet(meta_path, index=False, engine="pyarrow")
print(f"✅ Metadata saved: {meta_path}")

# Save feature column names for downstream loading
import json
cols_path = ARTIFACT_DIR / "feature_columns.json"
with open(cols_path, "w") as f:
    json.dump({"morgan_cols": morgan_cols, "rdkit_cols": rdkit_cols,
               "all_cols": feature_cols,
               "morgan_nbits": MORGAN_NBITS, "morgan_radius": MORGAN_RADIUS,
               "n_rdkit_2d": len(RDKIT_2D_DESCRIPTORS)}, f, indent=2)
print(f"✅ Column names saved: {cols_path}")

print(f"\n  → Next: step3_scaffold_splitting.py")
print("=" * 65)
