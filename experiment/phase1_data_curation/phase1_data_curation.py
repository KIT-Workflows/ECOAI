# ================================================================
# PHASE 1 — DATA CURATION PIPELINE
# ================================================================
# Reproducible pipeline that reads three SDF files, standardizes
# molecules, assigns labels, deduplicates by InChIKey, computes
# Bemis–Murcko scaffolds, flags malformed records, and exports
# a single curated_molecules.parquet file.
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# Uncomment the line below and run ONCE to install all packages.
# If using Anaconda, install rdkit via conda first:
#     conda install -c conda-forge rdkit
# Then pip-install the rest:

# !pip install rdkit-pypi pandas pyarrow numpy tqdm


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import os
import json
import hashlib
import warnings
from pathlib import Path
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── RDKit ──────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdMolDescriptors,
    SaltRemover,
    inchi as rdInchi,
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

# Suppress noisy RDKit warnings (set to WARNING for debugging)
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATASET_DIR  = PROJECT_ROOT / "Datasets"
DATA_OUT_DIR = DATASET_DIR / "data"
PHASE1_DIR   = PROJECT_ROOT / "experiment" / "phase1_data_curation"

# ── Input SDF Files ────────────────────────────────────────
SDF_PATHS = {
    "repellent":     DATASET_DIR / "repellent_library_hits_new_376_compounds.sdf",
    "non_repellent": DATASET_DIR / "non_repellent_library_decoys_376_compounds.sdf",
    "insecticide":   DATASET_DIR / "insecticide_library_LifeChemicals.sdf",
}

# ── Output Files ───────────────────────────────────────────
OUTPUT_PARQUET      = DATA_OUT_DIR / "curated_molecules.parquet"
OUTPUT_CURATION_LOG = PHASE1_DIR  / "curation_log.json"

# ── ID Prefixes per source ─────────────────────────────────
ID_PREFIX = {
    "repellent":     "REP",
    "non_repellent": "DEC",   # decoy
    "insecticide":   "LIC",   # LifeChemicals
}

# ── Reproducibility ───────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Verify inputs ─────────────────────────────────────────
for name, path in SDF_PATHS.items():
    assert path.exists(), f"❌ File not found: {path}"
for d in [DATA_OUT_DIR, PHASE1_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 1 — DATA CURATION PIPELINE")
print("=" * 60)
print(f"  Project root : {PROJECT_ROOT}")
print(f"  Output file  : {OUTPUT_PARQUET}")
print()
for name, path in SDF_PATHS.items():
    size_kb = path.stat().st_size / 1024
    print(f"  ✅ {name:15s} → {path.name}  ({size_kb:,.0f} KB)")
print()


# ================================================================
# Cell 2 — Standardization Helper Functions
# ================================================================
# These functions form the core curation pipeline applied to
# every molecule: sanitize → strip salts → neutralize → canonicalize.

# Pre-build reusable standardization objects (created once, used many times)
_salt_remover       = SaltRemover.SaltRemover()
_largest_frag       = rdMolStandardize.LargestFragmentChooser()
_uncharger          = rdMolStandardize.Uncharger()


def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Full standardization pipeline for a single RDKit Mol object.

    Steps:
        1. Remove explicit Hs
        2. Keep largest fragment (removes salts/counterions)
        3. Neutralize charges where chemically sensible
        4. Re-sanitize

    Returns None if any step fails.
    """
    if mol is None:
        return None
    try:
        # Step 1: Remove explicit hydrogens
        mol = Chem.RemoveHs(mol)

        # Step 2: Keep largest fragment (salt/counterion stripping)
        mol = _largest_frag.choose(mol)

        # Step 3: Neutralize charges
        mol = _uncharger.uncharge(mol)

        # Step 4: Re-sanitize to ensure consistency
        Chem.SanitizeMol(mol)

        return mol
    except Exception:
        return None


def mol_to_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    """Generate canonical SMILES string from an RDKit Mol."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def mol_to_inchikey(mol: Chem.Mol) -> Optional[str]:
    """Generate InChIKey from an RDKit Mol. Returns None on failure."""
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
    """
    Compute the generic Bemis–Murcko scaffold SMILES.

    Uses the generic scaffold (all atoms → carbon, all bonds → single)
    so that structurally similar molecules share the same scaffold
    regardless of heteroatom decoration.
    """
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        smi = Chem.MolToSmiles(generic, canonical=True)
        return smi if smi else None  # acyclic molecules → None
    except Exception:
        return None


def compute_qc_status(
    original_mol: Optional[Chem.Mol],
    std_mol: Optional[Chem.Mol],
    source: str,
) -> str:
    """
    Assign a QC status string to each molecule.

    Statuses:
        'pass'                 — molecule is fine for training/screening
        'parse_failed'         — RDKit could not parse the mol block
        'standardization_failed' — standardization pipeline returned None
        'too_small'            — fewer than 3 heavy atoms after stripping
        'disconnected'         — still has disconnected fragments
        'no_smiles'            — canonical SMILES generation failed
    """
    # Could not parse
    if original_mol is None:
        return "parse_failed"

    # Standardization failed
    if std_mol is None:
        return "standardization_failed"

    try:
        # Too small
        num_heavy = std_mol.GetNumHeavyAtoms()
        if num_heavy < 3:
            return "too_small"

        # Still disconnected (multiple fragments remain)
        smiles = Chem.MolToSmiles(std_mol)
        if smiles and "." in smiles:
            return "disconnected"

        # SMILES generation check
        if not smiles:
            return "no_smiles"

        return "pass"

    except Exception:
        return "standardization_failed"


def mol_to_formula(mol: Chem.Mol) -> Optional[str]:
    """Molecular formula string."""
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


print("✅ Standardization functions defined.")


# ================================================================
# Cell 3 — SDF Reader Function
# ================================================================
# Generic reader that parses any SDF, applies the standardization
# pipeline, extracts all SDF properties, and returns a list of
# record dicts ready for DataFrame construction.

def read_and_curate_sdf(
    sdf_path: Path,
    source_name: str,
    id_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Read an SDF file, standardize every molecule, and return
    a list of record dicts.

    Parameters
    ----------
    sdf_path : Path
        Path to the .sdf file.
    source_name : str
        Label for the source dataset (e.g. 'repellent').
    id_prefix : str
        Prefix for compound IDs (e.g. 'REP').

    Returns
    -------
    list of dict
        One dict per molecule with all computed fields and
        any SDF properties found in the file.
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    records = []
    sdf_property_names = set()

    print(f"\n── Reading: {sdf_path.name} ({source_name}) ──")

    for idx, raw_mol in enumerate(tqdm(supplier, desc=f"  {source_name}")):

        # ── Unique compound ID ──
        compound_id = f"{id_prefix}_{idx + 1:05d}"

        # ── Attempt to sanitize the raw mol ──
        original_mol = None
        if raw_mol is not None:
            try:
                Chem.SanitizeMol(raw_mol)
                original_mol = raw_mol
            except Exception:
                original_mol = None

        # ── Extract SDF properties BEFORE standardization ──
        sdf_props = {}
        if raw_mol is not None:
            try:
                for prop_name in raw_mol.GetPropsAsDict():
                    sdf_props[f"sdf_{prop_name}"] = raw_mol.GetPropsAsDict()[prop_name]
                    sdf_property_names.add(f"sdf_{prop_name}")
            except Exception:
                pass

        # ── Molecule name from SDF ──
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

        # ── Standardize ──
        std_mol = standardize_mol(original_mol)

        # ── Compute fields ──
        canonical_smiles = mol_to_canonical_smiles(std_mol)
        inchikey         = mol_to_inchikey(std_mol)
        formula          = mol_to_formula(std_mol)
        scaffold         = compute_scaffold(std_mol)
        qc_status        = compute_qc_status(original_mol, std_mol, source_name)

        # ── Basic descriptors ──
        num_heavy_atoms = std_mol.GetNumHeavyAtoms() if std_mol else None
        mol_weight      = round(Descriptors.ExactMolWt(std_mol), 4) if std_mol else None

        # ── Assemble record ──
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

    # ── Summary ──
    n_total  = len(records)
    n_pass   = sum(1 for r in records if r["qc_status"] == "pass")
    n_fail   = n_total - n_pass
    print(f"  → Parsed {n_total} molecules: {n_pass} pass, {n_fail} flagged")
    if sdf_property_names:
        print(f"  → SDF properties found: {sorted(sdf_property_names)}")

    return records


print("✅ SDF reader function defined.")


# ================================================================
# Cell 4 — Explore SDF Files (Field Discovery)
# ================================================================
# Quick read of each SDF to discover what metadata/properties
# are embedded. This helps us understand the data before full
# curation.

print("\n" + "=" * 60)
print("  SDF FIELD EXPLORATION")
print("=" * 60)

for source_name, sdf_path in SDF_PATHS.items():
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    n_mols = 0
    all_props = Counter()
    sample_props = {}

    for mol in supplier:
        n_mols += 1
        if mol is not None:
            try:
                props = mol.GetPropsAsDict()
                for k, v in props.items():
                    all_props[k] += 1
                    if k not in sample_props:
                        sample_props[k] = v
            except Exception:
                pass

    print(f"\n📂 {source_name} — {sdf_path.name}")
    print(f"   Molecules: {n_mols}")
    if all_props:
        print(f"   Properties ({len(all_props)} unique):")
        for prop_name, count in sorted(all_props.items()):
            sample_val = sample_props.get(prop_name, "N/A")
            # Truncate long values for display
            sample_str = str(sample_val)[:60]
            print(f"     • {prop_name:30s}  ({count:>5d} mols)  sample: {sample_str}")
    else:
        print("   Properties: (none found)")

print()


# ================================================================
# Cell 5 — Parse and Curate All Three SDF Files
# ================================================================
# Apply the full curation pipeline to each SDF and collect results.

all_records = []

for source_name, sdf_path in SDF_PATHS.items():
    prefix = ID_PREFIX[source_name]
    records = read_and_curate_sdf(sdf_path, source_name, prefix)
    all_records.extend(records)

print(f"\n✅ Total raw records parsed: {len(all_records)}")


# ================================================================
# Cell 6 — Assign Labels and Metadata
# ================================================================
# Build the DataFrame, assign repellent_active labels, and add
# assay/species metadata as required by the project plan.

df = pd.DataFrame(all_records)

# ── Assign supervised labels ───────────────────────────────
# repellent → 1, non_repellent → 0, insecticide → NaN (unlabeled)
label_map = {
    "repellent":     1,
    "non_repellent": 0,
    "insecticide":   np.nan,   # unlabeled screening pool
}
df["repellent_active"] = df["source_dataset"].map(label_map)

# ── Assay / species metadata ──────────────────────────────
# The first supervised endpoint targets Aedes aegypti repellency.
# Labeled compounds get assay metadata; unlabeled get NaN.
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

print(f"\n✅ Labels assigned.")
print(f"   Label distribution:")
print(f"     repellent_active = 1 : {(df['repellent_active'] == 1).sum()}")
print(f"     repellent_active = 0 : {(df['repellent_active'] == 0).sum()}")
print(f"     unlabeled (NaN)      : {df['repellent_active'].isna().sum()}")
print(f"\n   Source distribution:")
print(df["source_dataset"].value_counts().to_string(header=False))


# ================================================================
# Cell 7 — Deduplicate by InChIKey
# ================================================================
# Keep one canonical record per InChIKey. If the same molecule
# appears in multiple sources, we flag the conflict.

print("\n" + "=" * 60)
print("  DEDUPLICATION")
print("=" * 60)

n_before = len(df)

# Separate molecules that have vs. don't have an InChIKey
df_has_key = df[df["inchikey"].notna()].copy()
df_no_key  = df[df["inchikey"].isna()].copy()

print(f"  Molecules with InChIKey:    {len(df_has_key)}")
print(f"  Molecules without InChIKey: {len(df_no_key)} (kept but flagged)")

# ── Check for cross-source duplicates ──────────────────────
cross_source_dupes = (
    df_has_key.groupby("inchikey")["source_dataset"]
    .nunique()
    .loc[lambda x: x > 1]
)
if len(cross_source_dupes) > 0:
    print(f"\n  ⚠️  {len(cross_source_dupes)} InChIKeys appear in multiple sources!")
    # Show details
    for ik in cross_source_dupes.index[:10]:  # show first 10
        sources = df_has_key.loc[df_has_key["inchikey"] == ik, "source_dataset"].tolist()
        print(f"     {ik}  → {sources}")
    if len(cross_source_dupes) > 10:
        print(f"     ... and {len(cross_source_dupes) - 10} more")

    # Flag cross-source duplicates
    df_has_key["_cross_source_dup"] = df_has_key["inchikey"].isin(cross_source_dupes.index)
else:
    df_has_key["_cross_source_dup"] = False
    print("  ✅ No cross-source duplicates found.")

# ── Deduplicate: keep first occurrence per InChIKey ────────
# Priority order: repellent > non_repellent > insecticide
# (so labeled data is preferred if a molecule appears in multiple sources)
source_priority = {"repellent": 0, "non_repellent": 1, "insecticide": 2}
df_has_key["_source_priority"] = df_has_key["source_dataset"].map(source_priority)
df_has_key = df_has_key.sort_values(
    ["inchikey", "_source_priority", "compound_id"]
).reset_index(drop=True)

n_dupes_within = df_has_key.duplicated(subset="inchikey", keep="first").sum()
df_deduped = df_has_key.drop_duplicates(subset="inchikey", keep="first").copy()
df_deduped = df_deduped.drop(columns=["_source_priority", "_cross_source_dup"])

# ── Re-combine with no-key molecules ──────────────────────
df = pd.concat([df_deduped, df_no_key], ignore_index=True)

n_after = len(df)
print(f"\n  Duplicates removed:  {n_dupes_within}")
print(f"  Before dedup: {n_before}  →  After dedup: {n_after}")


# ================================================================
# Cell 8 — Final QC Flagging and Status Summary
# ================================================================
# Update QC status for molecules that failed various checks.
# Molecules with qc_status != 'pass' should be EXCLUDED from
# supervised training but are kept in the parquet for traceability.

print("\n" + "=" * 60)
print("  QC STATUS SUMMARY")
print("=" * 60)

# ── Additional QC: flag molecules without InChIKey ─────────
mask_no_key = df["inchikey"].isna() & (df["qc_status"] == "pass")
df.loc[mask_no_key, "qc_status"] = "no_inchikey"

# ── Summary ────────────────────────────────────────────────
qc_summary = df.groupby(["source_dataset", "qc_status"]).size().unstack(fill_value=0)
print(qc_summary)
print()

# Trainable molecules (pass QC and have labels)
df_trainable = df[(df["qc_status"] == "pass") & (df["repellent_active"].notna())]
print(f"  Trainable molecules (pass QC + labeled): {len(df_trainable)}")
print(f"    repellent_active = 1 : {(df_trainable['repellent_active'] == 1).sum()}")
print(f"    repellent_active = 0 : {(df_trainable['repellent_active'] == 0).sum()}")


# ================================================================
# Cell 9 — Scaffold Statistics
# ================================================================
# Examine scaffold diversity among trainable molecules.

print("\n" + "=" * 60)
print("  SCAFFOLD STATISTICS")
print("=" * 60)

df_with_scaffold = df[(df["qc_status"] == "pass") & (df["scaffold_smiles"].notna())]

# Overall scaffold counts
n_unique_scaffolds = df_with_scaffold["scaffold_smiles"].nunique()
print(f"  Unique generic scaffolds (all pass molecules): {n_unique_scaffolds}")

# Scaffold counts by source
for src in ["repellent", "non_repellent", "insecticide"]:
    subset = df_with_scaffold[df_with_scaffold["source_dataset"] == src]
    n_sc = subset["scaffold_smiles"].nunique()
    n_mol = len(subset)
    print(f"    {src:15s}: {n_sc:>4d} scaffolds across {n_mol} molecules")

# Top 10 most common scaffolds among labeled data
labeled_scaffolds = (
    df_with_scaffold[df_with_scaffold["repellent_active"].notna()]
    ["scaffold_smiles"]
    .value_counts()
    .head(10)
)
print(f"\n  Top 10 scaffolds (labeled data):")
for scaffold, count in labeled_scaffolds.items():
    print(f"    [{count:>3d} mols]  {scaffold}")


# ================================================================
# Cell 10 — Assemble Final Column Order and Clean Up
# ================================================================
# Organize columns in a logical order and drop internal helpers.

# ── Define canonical column order ──────────────────────────
core_columns = [
    "compound_id",
    "mol_name",
    "canonical_smiles",
    "inchikey",
    "mol_formula",
    "mol_weight",
    "num_heavy_atoms",
    "source_dataset",
    "repellent_active",
    "target_species",
    "assay_type",
    "scaffold_smiles",
    "qc_status",
]

# Any remaining columns are SDF metadata (prefixed with sdf_)
sdf_columns = sorted([c for c in df.columns if c.startswith("sdf_")])
all_columns = core_columns + sdf_columns

# Keep only columns that actually exist in df
final_columns = [c for c in all_columns if c in df.columns]

# Also keep any columns we might have missed
extra_columns = [c for c in df.columns if c not in final_columns]
if extra_columns:
    print(f"  ℹ️  Extra columns not in standard order: {extra_columns}")
    # Drop internal helper columns (those starting with _)
    extra_columns = [c for c in extra_columns if not c.startswith("_")]
    final_columns.extend(extra_columns)

df = df[final_columns].copy()

# ── Sort for deterministic output ─────────────────────────
# Sort by source priority, then compound_id
source_sort_order = {"repellent": 0, "non_repellent": 1, "insecticide": 2}
df["_sort_key"] = df["source_dataset"].map(source_sort_order)
df = df.sort_values(["_sort_key", "compound_id"]).reset_index(drop=True)
df = df.drop(columns=["_sort_key"])

print(f"\n✅ Final DataFrame assembled: {df.shape[0]} molecules × {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")


# ================================================================
# Cell 11 — Save curated_molecules.parquet
# ================================================================
# Save the curated dataset as a Parquet file for efficient
# downstream loading. Also save a backup CSV for inspection.

# Fix mixed-type SDF columns (int/str/float) → cast all to string
# PyArrow cannot serialize columns with heterogeneous Python types.
sdf_cols = [c for c in df.columns if c.startswith("sdf_")]
for col in sdf_cols:
    df[col] = df[col].astype(str).replace("nan", pd.NA)

df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
print(f"\n✅ Saved: {OUTPUT_PARQUET}")
print(f"   Size: {OUTPUT_PARQUET.stat().st_size / 1024:.1f} KB")

# Also save a human-readable CSV for quick inspection
csv_path = PHASE1_DIR / "curated_molecules_preview.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Saved CSV preview: {csv_path}")


# ================================================================
# Cell 12 — Comprehensive Summary Statistics
# ================================================================
# Print a full summary of the curated dataset.

print("\n" + "=" * 60)
print("  CURATION SUMMARY")
print("=" * 60)

print(f"\n  📊 Total molecules in curated dataset: {len(df)}")
print(f"  📊 Columns: {len(df.columns)}")

print(f"\n  ── By Source ──")
for src, grp in df.groupby("source_dataset"):
    n = len(grp)
    n_pass = (grp["qc_status"] == "pass").sum()
    n_flag = n - n_pass
    print(f"    {src:15s}: {n:>5d} total  ({n_pass} pass, {n_flag} flagged)")

print(f"\n  ── By QC Status ──")
for status, count in df["qc_status"].value_counts().items():
    print(f"    {status:25s}: {count:>5d}")

print(f"\n  ── Label Distribution (among QC-pass only) ──")
df_pass = df[df["qc_status"] == "pass"]
print(f"    repellent_active = 1   : {(df_pass['repellent_active'] == 1).sum():>5d}")
print(f"    repellent_active = 0   : {(df_pass['repellent_active'] == 0).sum():>5d}")
print(f"    unlabeled (insecticide): {df_pass['repellent_active'].isna().sum():>5d}")

print(f"\n  ── Scaffold Diversity (QC-pass) ──")
print(f"    Unique scaffolds: {df_pass['scaffold_smiles'].nunique()}")

print(f"\n  ── Molecular Weight (QC-pass) ──")
mw = df_pass["mol_weight"].dropna()
print(f"    Min:    {mw.min():.1f}")
print(f"    Max:    {mw.max():.1f}")
print(f"    Mean:   {mw.mean():.1f}")
print(f"    Median: {mw.median():.1f}")

print(f"\n  ── Heavy Atom Count (QC-pass) ──")
ha = df_pass["num_heavy_atoms"].dropna()
print(f"    Min:    {ha.min():.0f}")
print(f"    Max:    {ha.max():.0f}")
print(f"    Mean:   {ha.mean():.1f}")
print(f"    Median: {ha.median():.0f}")


# ================================================================
# Cell 13 — Determinism Verification
# ================================================================
# Hash the output parquet to verify that re-runs produce identical
# results. Save the hash for future comparison.

def hash_file(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash digest of a file."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


parquet_hash = hash_file(OUTPUT_PARQUET)
print(f"\n  🔒 Output hash (SHA-256): {parquet_hash}")
print(f"     Re-run this pipeline — if the hash matches, curation is deterministic.")


# ================================================================
# Cell 14 — Export Curation Log (JSON)
# ================================================================
# Save a structured log of the curation run for reproducibility
# and downstream reference.

curation_log = {
    "pipeline":       "Phase 1 — Data Curation",
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
            "unlabeled":     int(df_pass["repellent_active"].isna().sum()),
        },
        "unique_scaffolds": int(df_pass["scaffold_smiles"].nunique()),
        "duplicates_removed": n_before - n_after,
    },
}

with open(OUTPUT_CURATION_LOG, "w") as f:
    json.dump(curation_log, f, indent=2)

print(f"\n✅ Curation log saved: {OUTPUT_CURATION_LOG}")


# ================================================================
# Cell 15 — Quick Data Preview
# ================================================================
# Display the first few rows of each source for visual inspection.

print("\n" + "=" * 60)
print("  DATA PREVIEW")
print("=" * 60)

for src in ["repellent", "non_repellent", "insecticide"]:
    subset = df[df["source_dataset"] == src].head(5)
    print(f"\n── {src} (first 5 rows) ──")
    print(subset[["compound_id", "canonical_smiles", "inchikey",
                   "mol_weight", "repellent_active", "qc_status"]].to_string(index=False))

print("\n" + "=" * 60)
print("  ✅ PHASE 1 COMPLETE — Data curation pipeline finished.")
print("=" * 60)
