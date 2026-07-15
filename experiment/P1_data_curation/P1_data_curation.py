# *PHASE 1 — DATA CURATION PIPELINE*
# 
# Reproducible pipeline that reads four SDF files, standardizes
# molecules, assigns labels (0=inactive, 1=repellent, 2=dual-active),
# deduplicates by InChIKey, computes Bemis–Murcko scaffolds, flags
# malformed records, and exports a single curated_molecules.parquet.
#
# Activity labels are extracted from the SDF 'Activity Label' property:
#   0 → inactive/non-repellent
#   1 → active repellent
#   2 → dual-active (repellent + insecticidal)
# 
# # Input:
# 1. G:/research/ECOAI/Datasets/ECOAI_repellent_dataset_Indexed_Literature+USDA_report.sdf    (381 repellent + 30 dual-active)
# 2. G:/research/ECOAI/Datasets/ECOAI_non_repellent_dataset_USDA_report.sdf                   (1,210 non-repellent)
# 3. G:/research/ECOAI/Datasets/ECOAI_insecticides_dataset_LifeChemicals.sdf                  (3,108 insecticides — screening target)
# 4. G:/research/ECOAI/Datasets/ECOAI_natural_products_dataset_EssOilDB.sdf                   (1,633 natural products — screening target)
# 
# # Output:
# 1. G:/research/ECOAI/experiment/P1_data_curation/artifacts/curated_molecules.parquet
# 2. G:/research/ECOAI/experiment/P1_data_curation/artifacts/curation_log.json
# 3. G:/research/ECOAI/experiment/P1_data_curation/artifacts/curated_molecules_preview.csv
# 4. G:/research/ECOAI/experiment/P1_data_curation/plots/P1_qc_distribution.png
# 5. G:/research/ECOAI/experiment/P1_data_curation/plots/P1_qc_distribution.jpg


# ================================================================
# S1 — Install Dependencies
# ================================================================
# Uncomment the line below and run ONCE to install all packages.
# !pip install rdkit-pypi pandas pyarrow numpy tqdm matplotlib


# ================================================================
# S2 — Imports and Configuration
# ================================================================
import os
import json
import hashlib
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    SaltRemover,
    inchi as rdInchi,
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

# Suppress rdkit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATASET_DIR  = PROJECT_ROOT / "Datasets"
P1_DIR       = PROJECT_ROOT / "experiment" / "P1_data_curation"
ARTIFACTS_DIR = P1_DIR / "artifacts"
PLOTS_DIR     = P1_DIR / "plots"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input SDF Files
SDF_PATHS = {
    "repellent":       DATASET_DIR / "ECOAI_repellent_dataset_Indexed_Literature+USDA_report.sdf",
    "non_repellent":   DATASET_DIR / "ECOAI_non_repellent_dataset_USDA_report.sdf",
    "insecticide":     DATASET_DIR / "ECOAI_insecticides_dataset_LifeChemicals.sdf",
    "natural_product": DATASET_DIR / "ECOAI_natural_products_dataset_EssOilDB.sdf",
}

# Output Files
OUTPUT_PARQUET      = ARTIFACTS_DIR / "curated_molecules.parquet"
OUTPUT_CURATION_LOG = ARTIFACTS_DIR / "curation_log.json"
OUTPUT_CSV_PREVIEW  = ARTIFACTS_DIR / "curated_molecules_preview.csv"

# ID Prefixes per source
ID_PREFIX = {
    "repellent":       "REP",
    "non_repellent":   "DEC",
    "insecticide":     "LIC",
    "natural_product": "NP",
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Verify inputs
for name, path in SDF_PATHS.items():
    assert path.exists(), f"[ERROR] File not found: {path}"

print("=" * 60)
print("  P1 — DATA CURATION PIPELINE")
print("=" * 60)
print(f"  Project root : {PROJECT_ROOT}")
print(f"  Output file  : {OUTPUT_PARQUET}")
print()
for name, path in SDF_PATHS.items():
    size_kb = path.stat().st_size / 1024
    print(f"  [OK] {name:15s} -> {path.name}  ({size_kb:,.0f} KB)")
print()


# ================================================================
# S3 — Standardization Helper Functions
# ================================================================
_salt_remover       = SaltRemover.SaltRemover()
_largest_frag       = rdMolStandardize.LargestFragmentChooser()
_uncharger          = rdMolStandardize.Uncharger()

def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Sanitize, strip salts/counterions, neutralize, re-sanitize."""
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
        mol = _largest_frag.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

def mol_to_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    """Generate canonical SMILES string from RDKit Mol."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None

def mol_to_inchikey(mol: Chem.Mol) -> Optional[str]:
    """Generate InChIKey from RDKit Mol."""
    if mol is None:
        return None
    try:
        inchi = rdInchi.MolToInchi(mol)
        if inchi is None:
            return None
        return rdInchi.InchiToInchiKey(inchi)
    except Exception:
        return None

def compute_scaffold(mol: Chem.Mol) -> Optional[str]:
    """Compute generic Bemis-Murcko scaffold SMILES (all single bonds/carbons)."""
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        smi = Chem.MolToSmiles(generic, canonical=True)
        return smi if smi else None
    except Exception:
        return None

def compute_qc_status(original_mol: Optional[Chem.Mol], std_mol: Optional[Chem.Mol]) -> str:
    """Assign QC status to molecule."""
    if original_mol is None:
        return "parse_failed"
    if std_mol is None:
        return "standardization_failed"
    try:
        num_heavy = std_mol.GetNumHeavyAtoms()
        if num_heavy < 3:
            return "too_small"
        smiles = Chem.MolToSmiles(std_mol)
        if smiles and "." in smiles:
            return "disconnected"
        if not smiles:
            return "no_smiles"
        return "pass"
    except Exception:
        return "standardization_failed"

def mol_to_formula(mol: Chem.Mol) -> Optional[str]:
    """Compute molecular formula string."""
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None

print("[OK] Standardization functions defined.")


# ================================================================
# S4 — SDF Reader Function
# ================================================================
def read_and_curate_sdf(
    sdf_path: Path,
    source_name: str,
    id_prefix: str,
) -> List[Dict[str, Any]]:
    """Read SDF file, standardize molecules, extract properties, and return record dicts."""
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    records = []
    sdf_property_names = set()

    print(f"\n--- Reading: {sdf_path.name} ({source_name}) ---")

    for idx, raw_mol in enumerate(tqdm(supplier, desc=f"  {source_name}")):
        compound_id = f"{id_prefix}_{idx + 1:05d}"
        original_mol = None
        if raw_mol is not None:
            try:
                Chem.SanitizeMol(raw_mol)
                original_mol = raw_mol
            except Exception:
                original_mol = None

        sdf_props = {}
        if raw_mol is not None:
            try:
                for prop_name in raw_mol.GetPropsAsDict():
                    sdf_props[f"sdf_{prop_name}"] = raw_mol.GetPropsAsDict()[prop_name]
                    sdf_property_names.add(f"sdf_{prop_name}")
            except Exception:
                pass

        mol_name = None
        if raw_mol is not None:
            try:
                mol_name = raw_mol.GetProp("_Name")
                if mol_name:
                    mol_name = mol_name.strip()
                if not mol_name:
                    mol_name = None
            except Exception:
                mol_name = None

        std_mol = standardize_mol(original_mol)
        canonical_smiles = mol_to_canonical_smiles(std_mol)
        inchikey         = mol_to_inchikey(std_mol)
        formula          = mol_to_formula(std_mol)
        scaffold         = compute_scaffold(std_mol)
        qc_status        = compute_qc_status(original_mol, std_mol)

        num_heavy_atoms = std_mol.GetNumHeavyAtoms() if std_mol else None
        mol_weight      = round(Descriptors.ExactMolWt(std_mol), 4) if std_mol else None

        record = {
            "compound_id":      compound_id,
            "mol_name":         mol_name,
            "canonical_smiles": canonical_smiles,
            "inchikey":         inchikey,
            "mol_formula":      formula,
            "mol_weight":       mol_weight,
            "num_heavy_atoms":  num_heavy_atoms,
            "source_dataset":   source_name,
            "scaffold_smiles":  scaffold,
            "qc_status":        qc_status,
        }
        record.update(sdf_props)
        records.append(record)

    n_total  = len(records)
    n_pass   = sum(1 for r in records if r["qc_status"] == "pass")
    n_fail   = n_total - n_pass
    print(f"  -> Parsed {n_total} molecules: {n_pass} pass, {n_fail} flagged")

    return records

print("[OK] SDF reader function defined.")


# ================================================================
# S5 — Parse and Curate All Three SDF Files
# ================================================================
all_records = []
for source_name, sdf_path in SDF_PATHS.items():
    prefix = ID_PREFIX[source_name]
    records = read_and_curate_sdf(sdf_path, source_name, prefix)
    all_records.extend(records)

print(f"\n[OK] Total raw records parsed: {len(all_records)}")


# ================================================================
# S6 — Assign Labels and Metadata
# ================================================================
df = pd.DataFrame(all_records)

# Extract raw activity_class from SDF 'Activity Label' property
# New datasets use: 0=inactive/non-repellent, 1=active/repellent, 2=dual-active
if "sdf_Activity Label" in df.columns:
    df["activity_class"] = pd.to_numeric(df["sdf_Activity Label"], errors="coerce")
else:
    df["activity_class"] = np.nan

# Assign repellent_active label:
#   repellent SDF:       activity_class 1 or 2 → repellent_active = 1
#   non_repellent SDF:   activity_class 0      → repellent_active = 0
#   insecticide SDF:     unlabeled (NaN) — screening target only
#   natural_product SDF: unlabeled (NaN) — screening target only
label_map = {
    "repellent":       1,
    "non_repellent":   0,
    "insecticide":     np.nan,
    "natural_product": np.nan,
}
df["repellent_active"] = df["source_dataset"].map(label_map)

# Use SDF-provided Target Organism if available, else default
if "sdf_Target Organism" in df.columns:
    df["target_species"] = df["sdf_Target Organism"].replace(
        {"Not applied": np.nan, "": np.nan}
    )
else:
    df["target_species"] = np.where(
        df["source_dataset"].isin(["repellent", "non_repellent"]),
        "Aedes aegypti",
        np.nan,
    )
df["assay_type"] = np.where(
    df["source_dataset"].isin(["repellent", "non_repellent"]),
    "repellency",
    np.nan,
)

# --- Class Imbalance Report ---
n_pos = int((df["repellent_active"] == 1).sum())
n_neg = int((df["repellent_active"] == 0).sum())
n_dual = int((df["activity_class"] == 2).sum())
imbalance_ratio = n_neg / n_pos if n_pos > 0 else float("inf")

print(f"\n[OK] Labels assigned.")
print(f"   Label distribution:")
print(f"     repellent_active = 1 : {n_pos}  (incl. {n_dual} dual-active)")
print(f"     repellent_active = 0 : {n_neg}")
print(f"     unlabeled (NaN)      : {df['repellent_active'].isna().sum()}")
print(f"     Class imbalance ratio: {imbalance_ratio:.2f}:1 (neg:pos)")
if imbalance_ratio > 2:
    print(f"     [WARN] Significant class imbalance — use class_weight='balanced' in models.")


# ================================================================
# S7 — Deduplicate by InChIKey
# ================================================================
print("\n" + "=" * 60)
print("  DEDUPLICATION")
print("=" * 60)

n_before = len(df)
df_has_key = df[df["inchikey"].notna()].copy()
df_no_key  = df[df["inchikey"].isna()].copy()

cross_source_dupes = (
    df_has_key.groupby("inchikey")["source_dataset"]
    .nunique()
    .loc[lambda x: x > 1]
)
if len(cross_source_dupes) > 0:
    print(f"\n  [WARNING] {len(cross_source_dupes)} InChIKeys appear in multiple sources!")
    df_has_key["_cross_source_dup"] = df_has_key["inchikey"].isin(cross_source_dupes.index)
else:
    df_has_key["_cross_source_dup"] = False
    print("  [OK] No cross-source duplicates found.")

source_priority = {"repellent": 0, "non_repellent": 1, "insecticide": 2, "natural_product": 3}
df_has_key["_source_priority"] = df_has_key["source_dataset"].map(source_priority)
df_has_key = df_has_key.sort_values(
    ["inchikey", "_source_priority", "compound_id"]
).reset_index(drop=True)

n_dupes_within = df_has_key.duplicated(subset="inchikey", keep="first").sum()
df_deduped = df_has_key.drop_duplicates(subset="inchikey", keep="first").copy()
df_deduped = df_deduped.drop(columns=["_source_priority", "_cross_source_dup"])

df = pd.concat([df_deduped, df_no_key], ignore_index=True)
n_after = len(df)
print(f"\n  Duplicates removed:  {n_dupes_within}")
print(f"  Before dedup: {n_before}  ->  After dedup: {n_after}")


# ================================================================
# S8 — Final QC Flagging and Status Summary
# ================================================================
print("\n" + "=" * 60)
print("  QC STATUS SUMMARY")
print("=" * 60)

mask_no_key = df["inchikey"].isna() & (df["qc_status"] == "pass")
df.loc[mask_no_key, "qc_status"] = "no_inchikey"

qc_summary = df.groupby(["source_dataset", "qc_status"]).size().unstack(fill_value=0)
print(qc_summary)
print()

df_pass = df[df["qc_status"] == "pass"]
df_trainable = df[(df["qc_status"] == "pass") & (df["repellent_active"].notna())]
print(f"  Trainable molecules (pass QC + labeled): {len(df_trainable)}")


# ================================================================
# S9 — Assemble Final Column Order and Clean Up
# ================================================================
core_columns = [
    "compound_id",
    "mol_name",
    "canonical_smiles",
    "inchikey",
    "mol_formula",
    "mol_weight",
    "num_heavy_atoms",
    "source_dataset",
    "activity_class",
    "repellent_active",
    "target_species",
    "assay_type",
    "scaffold_smiles",
    "qc_status",
]
sdf_columns = sorted([c for c in df.columns if c.startswith("sdf_")])
all_columns = core_columns + sdf_columns
final_columns = [c for c in all_columns if c in df.columns]

df = df[final_columns].copy()

source_sort_order = {"repellent": 0, "non_repellent": 1, "insecticide": 2, "natural_product": 3}
df["_sort_key"] = df["source_dataset"].map(source_sort_order)
df = df.sort_values(["_sort_key", "compound_id"]).reset_index(drop=True)
df = df.drop(columns=["_sort_key"])


# ================================================================
# S10 — Save Curation Results
# ================================================================
sdf_cols = [c for c in df.columns if c.startswith("sdf_")]
for col in sdf_cols:
    df[col] = df[col].astype(str).replace("nan", pd.NA)

df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
print(f"\n[OK] Saved: {OUTPUT_PARQUET}")

df.to_csv(OUTPUT_CSV_PREVIEW, index=False)
print(f"[OK] Saved CSV preview: {OUTPUT_CSV_PREVIEW}")


# ================================================================
# S11 — Determinism Verification
# ================================================================
def hash_file(filepath: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

parquet_hash = hash_file(OUTPUT_PARQUET)
print(f"\n  [SECURE] Output hash (SHA-256): {parquet_hash}")


# ================================================================
# S12 — Generate QC Distribution Plot (PNG + JPG)
# ================================================================
qc_counts = df.groupby(["source_dataset", "qc_status"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
qc_counts.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="black", alpha=0.85)
ax.set_title("QC Status Distribution by Source Dataset", fontsize=14, fontweight="bold")
ax.set_xlabel("Source Dataset", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Molecules", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()

plot_path_png = PLOTS_DIR / "P1_qc_distribution.png"
plot_path_jpg = PLOTS_DIR / "P1_qc_distribution.jpg"

plt.savefig(plot_path_png, dpi=300)
plt.savefig(plot_path_jpg, dpi=300)
plt.close()

print(f"[OK] Saved plots:\n  - {plot_path_png}\n  - {plot_path_jpg}")


# ================================================================
# S13 — Export Curation Log (JSON)
# ================================================================
curation_log = {
    "pipeline":       "P1 — Data Curation",
    "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
    "random_seed":    RANDOM_SEED,
    "input_files": {
        name: {
            "path":     str(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in SDF_PATHS.items()
    },
    "output_file": {
        "path":      str(OUTPUT_PARQUET),
        "size_bytes": OUTPUT_PARQUET.stat().st_size,
        "sha256":    parquet_hash,
    },
    "counts": {
        "total_raw":    len(all_records),
        "total_curated": len(df),
        "by_source": df["source_dataset"].value_counts().to_dict(),
        "by_qc_status": df["qc_status"].value_counts().to_dict(),
        "labeled_pass": {
            "repellent":     int((df_pass["repellent_active"] == 1).sum()),
            "non_repellent": int((df_pass["repellent_active"] == 0).sum()),
            "dual_active":   int((df_pass["activity_class"] == 2).sum()) if "activity_class" in df_pass.columns else 0,
            "unlabeled":     int(df_pass["repellent_active"].isna().sum()),
        },
        "class_imbalance_ratio": round(imbalance_ratio, 2),
        "unique_scaffolds": int(df_pass["scaffold_smiles"].nunique()),
        "duplicates_removed": n_before - n_after,
    },
}

with open(OUTPUT_CURATION_LOG, "w") as f:
    json.dump(curation_log, f, indent=2)

print(f"[OK] Curation log saved: {OUTPUT_CURATION_LOG}")


# ================================================================
# S14 — Final Summary
# ================================================================
print("\n" + "=" * 60)
print("  P1 COMPLETE — Data Curation Pipeline")
print("=" * 60)
print(f"  Artifact Directory: {ARTIFACTS_DIR}")
print(f"  Plots Directory:    {PLOTS_DIR}")
print()
print(f"  Artifacts Generated:")
print(f"    1. {OUTPUT_PARQUET.name} (Curated parquet dataset)")
print(f"    2. {OUTPUT_CSV_PREVIEW.name} (Human-readable preview)")
print(f"    3. {OUTPUT_CURATION_LOG.name} (Pipeline execution details)")
print(f"    4. P1_qc_distribution.png / .jpg (QC status visualization)")
print()
print(f"  Summary:")
print(f"    - Total molecules parsed: {len(df)}")
print(f"    - Trainable molecules:    {len(df_trainable)}")
print(f"    - Unique scaffolds:       {df_pass['scaffold_smiles'].nunique()}")
print("=" * 60)
