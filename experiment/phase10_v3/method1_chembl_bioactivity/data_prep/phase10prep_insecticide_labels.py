# ================================================================
# PHASE 10 PREP — INSECTICIDE LABEL ACQUISITION FROM ChEMBL
# ================================================================
# Downloads insecticide bioactivity data from ChEMBL, standardizes
# structures using the Phase 1 pipeline, matches to our curated
# LifeChemicals library by InChIKey, binarizes activity labels,
# and exports the insecticide_labels.parquet artifact needed to
# activate Phase 10 (V3 Multitask).
#
# Strategy:
#   1. Query ChEMBL REST API for bioactivity against insect targets
#   2. Focus on mosquitoes (Aedes, Anopheles, Culex) + general pests
#   3. Also query key insecticide protein targets (AChE, GABA, nAChR)
#   4. Download compound SMILES + bioactivity values
#   5. Standardize structures (reuse Phase 1 standardization)
#   6. Match to LifeChemicals library by InChIKey
#   7. Binarize: pChEMBL >= 5 (≈ IC50 ≤ 10 μM) → active
#   8. Export insecticide_labels.parquet
#
# Input:  Datasets/data/curated_molecules.parquet (Phase 1 output)
# Output: Datasets/data/insecticide_labels.parquet
#         experiment/phase10prep_insecticide_labels/acquisition_report.json
#         experiment/phase10prep_insecticide_labels/chembl_raw_cache.parquet
#
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# Uncomment below to install:
# pip install chembl_webresource_client tqdm --quiet
# !pip install chembl_webresource_client requests rdkit-pypi pandas pyarrow numpy tqdm


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── RDKit (same standardization as Phase 1) ────────────────
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdMolDescriptors,
    inchi as rdInchi,
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT  = Path(r"G:\research\ECOAI")
DATASET_DIR   = PROJECT_ROOT / "Datasets"
DATA_DIR      = DATASET_DIR / "data"
PHASE10P_DIR  = PROJECT_ROOT / "experiment" / "phase10_v3" / "method1_chembl_bioactivity" / "data_prep"

# ── Input ──────────────────────────────────────────────────
INPUT_CURATED = DATA_DIR / "curated_molecules.parquet"

# ── Output ─────────────────────────────────────────────────
OUTPUT_LABELS     = DATA_DIR / "insecticide_labels.parquet"
RAW_CACHE         = PHASE10P_DIR / "chembl_raw_cache.parquet"
ACQUISITION_LOG   = PHASE10P_DIR / "acquisition_report.json"

# ── Binarization Thresholds ───────────────────────────────
# pChEMBL ≥ 5.0  →  IC50/EC50/LC50 ≤ 10 μM  →  active
# This is a standard cheminformatics threshold
PCHEMBL_ACTIVE_THRESHOLD  = 5.0
IC50_ACTIVE_THRESHOLD_NM  = 10_000   # 10 μM in nM
MORTALITY_ACTIVE_PCT      = 50.0     # ≥ 50% mortality → active

# ── ChEMBL API Configuration ──────────────────────────────
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_PAGE_SIZE = 1000   # max entries per API request
REQUEST_TIMEOUT  = 60     # seconds per request
REQUEST_DELAY    = 0.5    # seconds between requests (rate limit)

# ── Reproducibility ───────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Verify ─────────────────────────────────────────────────
assert INPUT_CURATED.exists(), f"❌ {INPUT_CURATED} — Run Phase 1 first!"
PHASE10P_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  PHASE 10 PREP — INSECTICIDE LABEL ACQUISITION")
print("=" * 65)
print(f"  Project root:  {PROJECT_ROOT}")
print(f"  Output:        {OUTPUT_LABELS}")
print(f"  Active if:     pChEMBL ≥ {PCHEMBL_ACTIVE_THRESHOLD} "
      f"(IC50 ≤ {IC50_ACTIVE_THRESHOLD_NM/1000:.0f} μM)")
print()


# ================================================================
# Cell 2 — Phase 1 Standardization Functions (Reused)
# ================================================================
# Exact same standardization as Phase 1 to ensure consistency.

_largest_frag = rdMolStandardize.LargestFragmentChooser()
_uncharger    = rdMolStandardize.Uncharger()


def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Phase 1 standardization: RemoveHs → largest frag → neutralize → sanitize."""
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


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """Convert SMILES → standardized InChIKey."""
    if not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = standardize_mol(mol)
        if mol is None:
            return None
        inchi = rdInchi.MolToInchi(mol)
        if inchi is None:
            return None
        return rdInchi.InchiToInchiKey(inchi)
    except Exception:
        return None


def smiles_to_canonical(smiles: str) -> Optional[str]:
    """Convert SMILES → standardized canonical SMILES."""
    if not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = standardize_mol(mol)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


print("✅ Phase 1 standardization functions loaded.")


# ================================================================
# Cell 3 — ChEMBL Query Engine (Two-Step: Targets → Activities)
# ================================================================
# ChEMBL does NOT allow filtering activities by target_organism
# directly. Instead we use a two-step approach:
#   Step 1: Find target ChEMBL IDs by organism name
#   Step 2: Query activities for each target ID
#
# Both the Python client and REST API follow this pattern.

USE_CLIENT_LIB = False

try:
    from chembl_webresource_client.new_client import new_client
    USE_CLIENT_LIB = True
    print("✅ chembl_webresource_client available — using Python client.")
except ImportError:
    print("⚠️  chembl_webresource_client not found — using REST API fallback.")

# REST API
import urllib.request
import urllib.error
import urllib.parse


def _chembl_rest_get(url: str) -> dict:
    """Single GET request to ChEMBL REST API, returns parsed JSON."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chembl_rest_paginate(endpoint: str, params: dict,
                          max_pages: int = 100) -> List[dict]:
    """
    Paginate through a ChEMBL REST endpoint.
    Handles URL-encoding properly for organism names with spaces.
    """
    results = []
    offset = 0

    for page in range(max_pages):
        # Properly URL-encode all parameter values
        encoded_params = urllib.parse.urlencode(params, safe=",")
        url = (f"{CHEMBL_API_BASE}/{endpoint}.json?"
               f"{encoded_params}"
               f"&limit={CHEMBL_PAGE_SIZE}&offset={offset}"
               f"&format=json")

        try:
            data = _chembl_rest_get(url)

            # ChEMBL returns results under a key matching the endpoint
            page_results = data.get(endpoint + "s", data.get("activities",
                                    data.get("targets", [])))
            if not page_results:
                # Try any list-valued key
                for key in data:
                    if isinstance(data[key], list) and len(data[key]) > 0:
                        page_results = data[key]
                        break

            if not page_results:
                break

            results.extend(page_results)

            # Check if there are more pages
            page_meta = data.get("page_meta", {})
            if page_meta.get("next") is None:
                break

            offset += CHEMBL_PAGE_SIZE
            time.sleep(REQUEST_DELAY)

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"    ⚠️  API error at offset {offset}: {e}")
            break
        except Exception as e:
            print(f"    ⚠️  Unexpected error: {e}")
            break

    return results


# ── Step 1 helpers: Find target IDs by organism ────────────


def _find_target_ids_rest(organism: str) -> List[str]:
    """REST: Find ChEMBL target IDs for an organism."""
    params = {"organism__icontains": organism}
    raw = _chembl_rest_paginate("target", params, max_pages=5)
    ids = [r["target_chembl_id"] for r in raw
           if r.get("target_chembl_id")]
    return list(set(ids))


def _find_target_ids_client(organism: str) -> List[str]:
    """Client lib: Find ChEMBL target IDs for an organism."""
    target_api = new_client.target
    results = target_api.filter(
        organism__icontains=organism
    ).only(["target_chembl_id"])
    ids = [r["target_chembl_id"] for r in results
           if r.get("target_chembl_id")]
    return list(set(ids))


def find_target_ids(organism: str) -> List[str]:
    """Find target IDs, trying client lib first, then REST."""
    if USE_CLIENT_LIB:
        try:
            return _find_target_ids_client(organism)
        except Exception as e:
            print(f"    ⚠️  Client target lookup failed: {e}")
            return _find_target_ids_rest(organism)
    return _find_target_ids_rest(organism)


# ── Step 2 helpers: Fetch activities by target ID ──────────

ACTIVITY_FIELDS = [
    "molecule_chembl_id", "canonical_smiles",
    "target_chembl_id", "target_organism",
    "assay_chembl_id", "assay_type",
    "standard_type", "standard_value",
    "standard_units", "standard_relation",
    "pchembl_value", "activity_comment",
]

STANDARD_TYPES = "IC50,EC50,LC50,LD50,GI50,Ki"


def _fetch_activities_rest(target_id: str) -> List[dict]:
    """REST: Fetch activity records for a single target ID."""
    params = {
        "target_chembl_id": target_id,
        "standard_type__in": STANDARD_TYPES,
    }
    return _chembl_rest_paginate("activity", params, max_pages=50)


def _fetch_activities_client(target_id: str) -> List[dict]:
    """Client lib: Fetch activity records for a single target ID."""
    activity_api = new_client.activity
    results = activity_api.filter(
        target_chembl_id=target_id,
        standard_type__in=STANDARD_TYPES.split(","),
    ).only(ACTIVITY_FIELDS)
    return list(results)


def fetch_activities(target_id: str) -> List[dict]:
    """Fetch activities, trying client lib first, then REST."""
    if USE_CLIENT_LIB:
        try:
            return _fetch_activities_client(target_id)
        except Exception as e:
            # Silent fallback to REST
            return _fetch_activities_rest(target_id)
    return _fetch_activities_rest(target_id)


# ── Combined query: organism → targets → activities ────────

def query_chembl(organism_filter: str) -> pd.DataFrame:
    """
    Two-step ChEMBL query:
      1. Find all target IDs for the organism
      2. Fetch all activities for those targets
    """
    print(f"  Querying: '{organism_filter}'...")

    # Step 1: Find targets
    target_ids = find_target_ids(organism_filter)
    if not target_ids:
        print(f"    → 0 targets found. Skipping.")
        return pd.DataFrame()
    print(f"    → {len(target_ids)} targets found. Fetching activities...")

    # Step 2: Fetch activities for each target
    all_records = []
    for i, tid in enumerate(target_ids):
        records = fetch_activities(tid)
        if records:
            all_records.extend(records)
        # Brief progress every 10 targets
        if (i + 1) % 10 == 0:
            print(f"      [{i+1}/{len(target_ids)}] "
                  f"{len(all_records)} records so far...")
        time.sleep(REQUEST_DELAY * 0.5)  # Lighter delay between target queries

    if not all_records:
        print(f"    → 0 activity records found.")
        return pd.DataFrame()

    # Extract to DataFrame with consistent columns
    rows = []
    for r in all_records:
        rows.append({field: r.get(field) for field in ACTIVITY_FIELDS})

    df = pd.DataFrame(rows)
    print(f"    → {len(df)} activity records found.")
    return df


print("✅ ChEMBL query engine ready (two-step: targets → activities).")


# ================================================================
# Cell 4 — Define Target Organisms & Insecticide Targets
# ================================================================
# We query ChEMBL for bioactivity against these organisms.
# Priority 1: Mosquitoes (most relevant to ECOAI project)
# Priority 2: Other important insect pests
# Priority 3: Insecticide protein targets (cross-species)

TARGET_ORGANISMS = [
    # ── Priority 1: Mosquitoes ──
    "Aedes aegypti",           # Yellow fever mosquito (primary ECOAI target)
    "Aedes albopictus",        # Asian tiger mosquito
    "Anopheles gambiae",       # African malaria mosquito
    "Anopheles stephensi",     # Asian malaria mosquito
    "Culex pipiens",           # Common house mosquito
    "Culex quinquefasciatus",  # Southern house mosquito

    # ── Priority 2: General insect pests ──
    "Musca domestica",         # Housefly
    "Drosophila melanogaster", # Fruit fly (model organism)
    "Spodoptera frugiperda",   # Fall armyworm
    "Spodoptera littoralis",   # African cotton leafworm
    "Plutella xylostella",     # Diamondback moth
    "Blattella germanica",     # German cockroach
    "Tribolium castaneum",     # Red flour beetle
    "Lucilia cuprina",         # Australian sheep blowfly
    "Haematobia irritans",     # Horn fly
    "Stomoxys calcitrans",     # Stable fly
]

# We also search broader terms to catch more data
BROAD_SEARCH_TERMS = [
    "mosquito",
    "insecticid",
]

print(f"  Target organisms: {len(TARGET_ORGANISMS)}")
print(f"  Broad search terms: {BROAD_SEARCH_TERMS}")


# ================================================================
# Cell 5 — Download Bioactivity Data from ChEMBL
# ================================================================
# Query all target organisms and collect results.
# Uses caching to avoid re-downloading on subsequent runs.

print("\n" + "=" * 65)
print("  DOWNLOADING ChEMBL BIOACTIVITY DATA")
print("=" * 65)

# Check for cached data first
if RAW_CACHE.exists():
    print(f"\n  📂 Found cached data: {RAW_CACHE}")
    print(f"     Loading from cache (delete file to re-download)...")
    df_raw = pd.read_parquet(RAW_CACHE)
    print(f"     → {len(df_raw)} cached records loaded.")
else:
    all_dfs = []

    # Query each organism
    for organism in TARGET_ORGANISMS:
        try:
            df_org = query_chembl(organism)
            if len(df_org) > 0:
                df_org["query_term"] = organism
                all_dfs.append(df_org)
        except Exception as e:
            print(f"  ⚠️  Failed for '{organism}': {e}")
        time.sleep(REQUEST_DELAY)

    # Broad search terms
    for term in BROAD_SEARCH_TERMS:
        try:
            df_term = query_chembl(term)
            if len(df_term) > 0:
                df_term["query_term"] = term
                all_dfs.append(df_term)
        except Exception as e:
            print(f"  ⚠️  Failed for '{term}': {e}")
        time.sleep(REQUEST_DELAY)

    if all_dfs:
        df_raw = pd.concat(all_dfs, ignore_index=True)

        # Deduplicate by (molecule_chembl_id, assay_chembl_id, standard_type)
        n_before = len(df_raw)
        df_raw = df_raw.drop_duplicates(
            subset=["molecule_chembl_id", "assay_chembl_id", "standard_type"],
            keep="first"
        )
        n_dupes = n_before - len(df_raw)

        print(f"\n  📊 Total raw records:    {n_before}")
        print(f"     Duplicates removed:   {n_dupes}")
        print(f"     Unique records:       {len(df_raw)}")

        # Cache for future runs
        df_raw.to_parquet(RAW_CACHE, index=False, engine="pyarrow")
        print(f"  💾 Cached to: {RAW_CACHE}")
    else:
        print("\n  ❌ No data retrieved from ChEMBL!")
        print("     Please check your internet connection.")
        print("     Alternatively, manually download data from:")
        print("       https://www.ebi.ac.uk/chembl/")
        df_raw = pd.DataFrame()


# ================================================================
# Cell 6 — Clean and Filter Activity Data
# ================================================================
# Keep only records with valid SMILES and quantitative activity.

print("\n" + "=" * 65)
print("  DATA CLEANING")
print("=" * 65)

if len(df_raw) == 0:
    print("  ⚠️  No raw data to clean. Cannot proceed.")
    print("     Re-run after resolving ChEMBL connectivity issues.")
else:
    n_raw = len(df_raw)

    # ── Drop records without SMILES ────────────────────────
    df_clean = df_raw[df_raw["canonical_smiles"].notna()].copy()
    n_no_smiles = n_raw - len(df_clean)
    print(f"  Dropped {n_no_smiles} records without SMILES.")

    # ── Convert numeric columns ────────────────────────────
    for col in ["standard_value", "pchembl_value"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # ── Keep only records with quantitative data ───────────
    has_pchembl = df_clean["pchembl_value"].notna()
    has_value   = df_clean["standard_value"].notna()
    df_clean = df_clean[has_pchembl | has_value].copy()
    n_no_quant = n_raw - n_no_smiles - len(df_clean)
    print(f"  Dropped {n_no_quant} records without quantitative data.")

    # ── Filter exact measurements only (= , not > or <) ───
    # Keep '=' relations and those without relation (assumed exact)
    valid_relations = ["=", "'='", None, np.nan, ""]
    mask_exact = (
        df_clean["standard_relation"].isna() |
        (df_clean["standard_relation"] == "=") |
        (df_clean["standard_relation"] == "'='")
    )
    n_approx = (~mask_exact).sum()
    df_clean = df_clean[mask_exact].copy()
    print(f"  Dropped {n_approx} records with approximate values (>, <, ~).")

    # ── Summary ────────────────────────────────────────────
    print(f"\n  📊 Clean dataset: {len(df_clean)} bioactivity records")
    print(f"     Unique molecules:   {df_clean['molecule_chembl_id'].nunique()}")
    print(f"     Target organisms:   {df_clean['target_organism'].nunique()}")
    print(f"     Assay types:        {df_clean['standard_type'].value_counts().to_dict()}")

    if df_clean["target_organism"].notna().any():
        print(f"\n     Top organisms:")
        for org, cnt in df_clean["target_organism"].value_counts().head(15).items():
            print(f"       {org:40s} {cnt:>6d}")


# ================================================================
# Cell 7 — Standardize Structures
# ================================================================
# Apply Phase 1 standardization to all ChEMBL compounds.
# Compute canonical SMILES and InChIKey for matching.

print("\n" + "=" * 65)
print("  STRUCTURE STANDARDIZATION")
print("=" * 65)

if len(df_clean) > 0:
    # Get unique SMILES to avoid redundant computation
    unique_smiles = df_clean["canonical_smiles"].unique()
    print(f"  Standardizing {len(unique_smiles)} unique SMILES...")

    smiles_map = {}  # original → {canonical_smiles, inchikey}

    for smi in tqdm(unique_smiles, desc="  Standardizing"):
        canonical = smiles_to_canonical(smi)
        inchikey  = smiles_to_inchikey(smi)
        smiles_map[smi] = {
            "std_canonical_smiles": canonical,
            "std_inchikey": inchikey,
        }

    # Map back to full dataframe
    df_clean["std_canonical_smiles"] = df_clean["canonical_smiles"].map(
        lambda x: smiles_map.get(x, {}).get("std_canonical_smiles")
    )
    df_clean["std_inchikey"] = df_clean["canonical_smiles"].map(
        lambda x: smiles_map.get(x, {}).get("std_inchikey")
    )

    # Drop failed standardizations
    n_before = len(df_clean)
    df_clean = df_clean[df_clean["std_inchikey"].notna()].copy()
    n_failed = n_before - len(df_clean)

    print(f"\n  ✅ Standardization complete.")
    print(f"     Succeeded: {len(df_clean)}")
    print(f"     Failed:    {n_failed}")
    print(f"     Unique InChIKeys: {df_clean['std_inchikey'].nunique()}")
else:
    print("  ⚠️  No data to standardize.")


# ================================================================
# Cell 8 — Binarize Activity Labels
# ================================================================
# Convert continuous activity values to binary insecticidal labels.
#
# Rules:
#   pChEMBL ≥ 5.0  → active (1)       [IC50 ≤ 10 μM]
#   pChEMBL < 5.0  → inactive (0)     [IC50 > 10 μM]
#
#   If pChEMBL not available:
#     standard_value ≤ 10000 nM → active (for IC50/EC50/Ki in nM)
#     standard_value ≤ 10 μM   → active (for IC50/EC50/Ki in μM)
#     standard_value ≤ 50 ppm  → active (for LC50 in ppm/mg/L)

print("\n" + "=" * 65)
print("  BINARIZATION OF ACTIVITY LABELS")
print("=" * 65)

if len(df_clean) > 0:

    def binarize_activity(row):
        """
        Convert a single bioactivity record to a binary label.
        Returns 1 (active), 0 (inactive), or NaN (ambiguous).
        """
        pchembl = row.get("pchembl_value")
        value   = row.get("standard_value")
        units   = str(row.get("standard_units", "")).strip().lower()
        stype   = str(row.get("standard_type", "")).strip().upper()

        # Strategy 1: Use pChEMBL (most reliable, already normalized)
        if pd.notna(pchembl):
            return 1 if pchembl >= PCHEMBL_ACTIVE_THRESHOLD else 0

        # Strategy 2: Use standard_value with unit awareness
        if pd.notna(value):
            if stype in ["IC50", "EC50", "GI50", "Ki"]:
                if units in ["nm", "nmol/l"]:
                    return 1 if value <= IC50_ACTIVE_THRESHOLD_NM else 0
                elif units in ["um", "umol/l", "µm"]:
                    return 1 if value <= (IC50_ACTIVE_THRESHOLD_NM / 1000) else 0
                elif units in ["mm", "mmol/l"]:
                    return 1 if value <= (IC50_ACTIVE_THRESHOLD_NM / 1e6) else 0
            elif stype in ["LC50", "LD50"]:
                if units in ["ug/ml", "mg/l", "ppm", "ug ml-1"]:
                    return 1 if value <= 50 else 0  # 50 ppm threshold
                elif units in ["nm", "nmol/l"]:
                    return 1 if value <= IC50_ACTIVE_THRESHOLD_NM else 0
                elif units in ["um", "umol/l", "µm"]:
                    return 1 if value <= (IC50_ACTIVE_THRESHOLD_NM / 1000) else 0

        return np.nan  # Cannot determine

    df_clean["insecticidal_active"] = df_clean.apply(binarize_activity, axis=1)

    # Drop ambiguous records
    n_total = len(df_clean)
    n_labeled = df_clean["insecticidal_active"].notna().sum()
    n_ambiguous = n_total - n_labeled

    print(f"  Total records:     {n_total}")
    print(f"  Successfully labeled: {n_labeled}")
    print(f"  Ambiguous (dropped):  {n_ambiguous}")

    df_labeled = df_clean[df_clean["insecticidal_active"].notna()].copy()
    df_labeled["insecticidal_active"] = df_labeled["insecticidal_active"].astype(int)

    print(f"\n  Label distribution:")
    print(f"    active   (1): {(df_labeled['insecticidal_active'] == 1).sum()}")
    print(f"    inactive (0): {(df_labeled['insecticidal_active'] == 0).sum()}")
else:
    df_labeled = pd.DataFrame()
    print("  ⚠️  No data to binarize.")


# ================================================================
# Cell 9 — Aggregate Per-Compound Labels
# ================================================================
# A single compound may have multiple bioactivity records.
# Aggregate to one label per compound by majority vote.
# If conflicting, use the median pChEMBL value to decide.

print("\n" + "=" * 65)
print("  PER-COMPOUND AGGREGATION")
print("=" * 65)

if len(df_labeled) > 0:
    # Group by standardized InChIKey
    compound_groups = df_labeled.groupby("std_inchikey")

    compound_records = []
    for inchikey, group in compound_groups:
        n_records = len(group)
        n_active  = (group["insecticidal_active"] == 1).sum()
        n_inactive = (group["insecticidal_active"] == 0).sum()

        # Majority vote
        if n_active > n_inactive:
            label = 1
        elif n_inactive > n_active:
            label = 0
        else:
            # Tie → use median pChEMBL
            med_pchembl = group["pchembl_value"].median()
            label = 1 if (pd.notna(med_pchembl) and med_pchembl >= PCHEMBL_ACTIVE_THRESHOLD) else 0

        # Representative data
        best_row = group.sort_values("pchembl_value", ascending=False).iloc[0]

        compound_records.append({
            "inchikey":             inchikey,
            "canonical_smiles":     best_row["std_canonical_smiles"],
            "molecule_chembl_id":   best_row["molecule_chembl_id"],
            "insecticidal_active":  label,
            "n_assay_records":      n_records,
            "n_active_records":     n_active,
            "n_inactive_records":   n_inactive,
            "best_pchembl":         group["pchembl_value"].max(),
            "median_pchembl":       group["pchembl_value"].median(),
            "assay_types":          ",".join(sorted(group["standard_type"].dropna().unique())),
            "target_organisms":     ",".join(sorted(group["target_organism"].dropna().unique())),
            "source":               "ChEMBL",
        })

    df_compounds = pd.DataFrame(compound_records)

    print(f"  Unique compounds: {len(df_compounds)}")
    print(f"  Label distribution:")
    print(f"    active   (1): {(df_compounds['insecticidal_active'] == 1).sum()}")
    print(f"    inactive (0): {(df_compounds['insecticidal_active'] == 0).sum()}")
    print(f"\n  Records per compound:")
    print(f"    Mean:   {df_compounds['n_assay_records'].mean():.1f}")
    print(f"    Median: {df_compounds['n_assay_records'].median():.0f}")
    print(f"    Max:    {df_compounds['n_assay_records'].max()}")
else:
    df_compounds = pd.DataFrame()
    print("  ⚠️  No labeled data to aggregate.")


# ================================================================
# Cell 10 — Match to LifeChemicals Library
# ================================================================
# Check which ChEMBL compounds also exist in our curated
# LifeChemicals library (from Phase 1). These get direct labels.
# Non-matching ChEMBL compounds are kept as external training data.

print("\n" + "=" * 65)
print("  MATCHING TO LIFECHEM LIBRARY")
print("=" * 65)

if len(df_compounds) > 0:
    # Load curated library
    df_curated = pd.read_parquet(INPUT_CURATED)
    df_lifechem = df_curated[df_curated["source_dataset"] == "insecticide"].copy()

    print(f"  LifeChemicals molecules: {len(df_lifechem)}")
    print(f"  ChEMBL compounds:       {len(df_compounds)}")

    # Match by InChIKey
    lifechem_inchikeys = set(df_lifechem["inchikey"].dropna().unique())
    df_compounds["in_lifechem"] = df_compounds["inchikey"].isin(lifechem_inchikeys)

    n_matched   = df_compounds["in_lifechem"].sum()
    n_external  = len(df_compounds) - n_matched

    print(f"\n  Matched to LifeChemicals:  {n_matched}")
    print(f"  External (ChEMBL only):    {n_external}")

    if n_matched > 0:
        matched = df_compounds[df_compounds["in_lifechem"]]
        print(f"\n  Matched label distribution:")
        print(f"    active   (1): {(matched['insecticidal_active'] == 1).sum()}")
        print(f"    inactive (0): {(matched['insecticidal_active'] == 0).sum()}")

    # Also try matching ALL curated molecules (not just insecticide source)
    all_inchikeys = set(df_curated["inchikey"].dropna().unique())
    df_compounds["in_curated_any"] = df_compounds["inchikey"].isin(all_inchikeys)
    n_matched_any = df_compounds["in_curated_any"].sum()
    if n_matched_any > n_matched:
        print(f"\n  ℹ️  Also found {n_matched_any - n_matched} matches in "
              f"repellent/non-repellent sources.")

    # Merge compound_id from curated data where possible
    curated_id_map = df_curated.set_index("inchikey")["compound_id"].to_dict()
    df_compounds["compound_id"] = df_compounds["inchikey"].map(curated_id_map)

    # For external compounds, assign new IDs
    ext_counter = 1
    for idx in df_compounds.index:
        if pd.isna(df_compounds.loc[idx, "compound_id"]):
            df_compounds.loc[idx, "compound_id"] = f"EXT_{ext_counter:05d}"
            ext_counter += 1

else:
    print("  ⚠️  No compounds to match.")


# ================================================================
# Cell 11 — Quality Control and Balancing
# ================================================================
# Check class balance and optionally subsample for balance.

print("\n" + "=" * 65)
print("  QUALITY CONTROL")
print("=" * 65)

if len(df_compounds) > 0:
    n_active   = (df_compounds["insecticidal_active"] == 1).sum()
    n_inactive = (df_compounds["insecticidal_active"] == 0).sum()
    total      = len(df_compounds)

    print(f"  Total compounds:  {total}")
    print(f"  Active (1):       {n_active} ({n_active/total*100:.1f}%)")
    print(f"  Inactive (0):     {n_inactive} ({n_inactive/total*100:.1f}%)")

    # Class balance ratio
    if min(n_active, n_inactive) > 0:
        imbalance = max(n_active, n_inactive) / min(n_active, n_inactive)
        print(f"  Imbalance ratio:  {imbalance:.1f}:1")
        if imbalance > 10:
            print(f"  ⚠️  Highly imbalanced — consider oversampling minority class.")
    else:
        print(f"  ⚠️  Only one class present! Cannot train a classifier.")

    # Summary by matching status
    print(f"\n  ── By Source ──")
    for source_type, label in [("LifeChemicals matched", True), ("External (ChEMBL)", False)]:
        subset = df_compounds[df_compounds["in_lifechem"] == label]
        if len(subset) > 0:
            n_a = (subset["insecticidal_active"] == 1).sum()
            n_i = (subset["insecticidal_active"] == 0).sum()
            print(f"    {source_type:25s}: {len(subset):>6d} "
                  f"({n_a} active, {n_i} inactive)")

    # pChEMBL distribution
    pchembl_vals = df_compounds["best_pchembl"].dropna()
    if len(pchembl_vals) > 0:
        print(f"\n  ── pChEMBL Distribution ──")
        print(f"    Min:    {pchembl_vals.min():.2f}")
        print(f"    Max:    {pchembl_vals.max():.2f}")
        print(f"    Mean:   {pchembl_vals.mean():.2f}")
        print(f"    Median: {pchembl_vals.median():.2f}")


# ================================================================
# Cell 12 — Assemble Final Label Table
# ================================================================
# Format output to match Phase 10 requirements.

print("\n" + "=" * 65)
print("  ASSEMBLING INSECTICIDE LABELS")
print("=" * 65)

if len(df_compounds) > 0:
    # Final column order per README spec
    output_columns = [
        "compound_id",
        "inchikey",
        "canonical_smiles",
        "insecticidal_active",
        "molecule_chembl_id",
        "n_assay_records",
        "n_active_records",
        "n_inactive_records",
        "best_pchembl",
        "median_pchembl",
        "assay_types",
        "target_organisms",
        "source",
        "in_lifechem",
        "in_curated_any",
    ]

    # Keep only columns that exist
    final_cols = [c for c in output_columns if c in df_compounds.columns]
    df_output = df_compounds[final_cols].copy()

    # Sort: LifeChemicals matches first, then by activity
    df_output = df_output.sort_values(
        ["in_lifechem", "insecticidal_active", "compound_id"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    print(f"  Final table: {df_output.shape[0]} compounds × {df_output.shape[1]} columns")
    print(f"  Columns: {list(df_output.columns)}")
else:
    df_output = pd.DataFrame()
    print("  ⚠️  No output to assemble.")


# ================================================================
# Cell 13 — Save insecticide_labels.parquet
# ================================================================

if len(df_output) > 0:
    df_output.to_parquet(OUTPUT_LABELS, index=False, engine="pyarrow")
    print(f"\n✅ Saved: {OUTPUT_LABELS}")
    print(f"   Size: {OUTPUT_LABELS.stat().st_size / 1024:.1f} KB")

    # Also save CSV for easy inspection
    csv_path = PHASE10P_DIR / "insecticide_labels_preview.csv"
    df_output.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")
else:
    print("\n  ❌ No data to save. Check ChEMBL connectivity.")


# ================================================================
# Cell 14 — Preview
# ================================================================

if len(df_output) > 0:
    print("\n" + "=" * 65)
    print("  DATA PREVIEW")
    print("=" * 65)

    # LifeChemicals matches
    matched = df_output[df_output["in_lifechem"] == True]
    if len(matched) > 0:
        print(f"\n  ── LifeChemicals Matches (first 10) ──")
        display_cols = ["compound_id", "insecticidal_active",
                       "best_pchembl", "assay_types", "target_organisms"]
        display_cols = [c for c in display_cols if c in matched.columns]
        print(matched[display_cols].head(10).to_string(index=False))

    # External compounds
    external = df_output[df_output["in_lifechem"] == False]
    if len(external) > 0:
        print(f"\n  ── External ChEMBL Compounds (first 10) ──")
        display_cols = ["compound_id", "canonical_smiles", "insecticidal_active",
                       "best_pchembl", "target_organisms"]
        display_cols = [c for c in display_cols if c in external.columns]
        print(external[display_cols].head(10).to_string(index=False))


# ================================================================
# Cell 15 — Save Acquisition Report
# ================================================================

report = {
    "pipeline":       "Phase 10 Prep — Insecticide Label Acquisition",
    "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
    "random_seed":    RANDOM_SEED,
    "data_source":    "ChEMBL",
    "query_method":   "chembl_webresource_client" if USE_CLIENT_LIB else "REST API",
    "target_organisms": TARGET_ORGANISMS,
    "broad_search_terms": BROAD_SEARCH_TERMS,
    "thresholds": {
        "pchembl_active":      PCHEMBL_ACTIVE_THRESHOLD,
        "ic50_active_nM":      IC50_ACTIVE_THRESHOLD_NM,
        "mortality_active_pct": MORTALITY_ACTIVE_PCT,
    },
    "counts": {
        "raw_records":          len(df_raw) if 'df_raw' in dir() else 0,
        "clean_records":        len(df_clean) if 'df_clean' in dir() else 0,
        "labeled_records":      len(df_labeled) if 'df_labeled' in dir() else 0,
        "unique_compounds":     len(df_compounds) if 'df_compounds' in dir() else 0,
        "insecticidal_active":  int((df_output["insecticidal_active"] == 1).sum()) if len(df_output) > 0 else 0,
        "insecticidal_inactive": int((df_output["insecticidal_active"] == 0).sum()) if len(df_output) > 0 else 0,
        "lifechem_matched":     int(df_output["in_lifechem"].sum()) if len(df_output) > 0 and "in_lifechem" in df_output.columns else 0,
        "external_compounds":   int((~df_output["in_lifechem"]).sum()) if len(df_output) > 0 and "in_lifechem" in df_output.columns else 0,
    },
    "output_files": {
        "labels_parquet": str(OUTPUT_LABELS),
        "raw_cache":      str(RAW_CACHE),
    },
    "next_step": "Activate Phase 10 (phase10_v3_multitask.py) with these labels.",
}

with open(ACQUISITION_LOG, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n✅ Acquisition report saved: {ACQUISITION_LOG}")


# ================================================================
# Cell 16 — Phase 10 Activation Readiness Check
# ================================================================

print("\n" + "=" * 65)
print("  PHASE 10 ACTIVATION READINESS")
print("=" * 65)

if len(df_output) > 0:
    n_active   = (df_output["insecticidal_active"] == 1).sum()
    n_inactive = (df_output["insecticidal_active"] == 0).sum()
    n_lifechem = df_output["in_lifechem"].sum() if "in_lifechem" in df_output.columns else 0

    checks = {
        "Labels file exists":           OUTPUT_LABELS.exists(),
        "Has active compounds (1)":     n_active > 0,
        "Has inactive compounds (0)":   n_inactive > 0,
        "Has both classes":             n_active > 0 and n_inactive > 0,
        "≥ 50 total compounds":         len(df_output) >= 50,
        "≥ 20 active compounds":        n_active >= 20,
        "≥ 20 inactive compounds":      n_inactive >= 20,
        "LifeChemicals matches exist":  n_lifechem > 0,
    }

    all_pass = True
    for check, passed in checks.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {check}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  🎉 ALL CHECKS PASS — Phase 10 can be activated!")
        print(f"     Run: experiment/phase10_v3_multitask/phase10_v3_multitask.py")
    else:
        print(f"\n  ⚠️  Some checks failed. You may need more data.")
        print(f"     Consider:")
        print(f"     • Expanding search to more organisms")
        print(f"     • Lowering activity threshold slightly")
        print(f"     • Adding PubChem BioAssay data as supplement")
else:
    print("  ❌ No labels produced. Cannot activate Phase 10.")


# ================================================================
# Cell 17 — Final Summary
# ================================================================

print("\n" + "=" * 65)
print("  PHASE 10 PREP COMPLETE — Insecticide Label Acquisition")
print("=" * 65)

if len(df_output) > 0:
    print(f"\n  📊 Results:")
    print(f"    • {len(df_output)} compounds with insecticidal labels")
    print(f"    • {(df_output['insecticidal_active'] == 1).sum()} active / "
          f"{(df_output['insecticidal_active'] == 0).sum()} inactive")
    n_lc = df_output["in_lifechem"].sum() if "in_lifechem" in df_output.columns else 0
    print(f"    • {n_lc} matched to LifeChemicals library")
    print(f"    • {len(df_output) - n_lc} external (ChEMBL-only) training compounds")

print(f"\n  📁 Artifacts:")
print(f"    1. {OUTPUT_LABELS}  (PHASE 10 INPUT)")
print(f"    2. {RAW_CACHE}")
print(f"    3. {ACQUISITION_LOG}")

print(f"\n  🔜 Next steps:")
print(f"    1. Review insecticide_labels_preview.csv")
print(f"    2. Activate Phase 10 (phase10_v3_multitask.py)")
print(f"    3. Train multitask model with both heads")
print(f"    4. Update predict_single_molecule.ipynb for 4-label output")
print("=" * 65)
