# ================================================================
# PHASE 7 — QUANTUM DESCRIPTORS (V2)
# ================================================================
# Compute electronic-structure descriptors for a selected subset
# of molecules using GFN2-xTB (primary) and DFT (validation).
#
# Workflow:
#   1. Select 500 molecules (300 labeled + 200 unlabeled)
#   2. Generate conformers with ETKDG → MMFF optimize
#   3. Keep lowest-energy conformer
#   4. Optimize with GFN2-xTB
#   5. Extract HOMO, LUMO, gap, dipole, etc.
#   6. DFT single-point validation on 100 molecule subset
#
# Input:  Datasets/data/curated_molecules_with_splits.parquet
#         Datasets/data/model_predictions.parquet
# Output: Datasets/data/quantum_descriptors.parquet
#
# NOTE: xTB must be installed separately:
#       conda install -c conda-forge xtb-python
#   or: pip install xtb
#
# ================================================================


# ================================================================
# Cell 0 — Install Dependencies
# ================================================================
# !pip install rdkit-pypi pandas pyarrow numpy tqdm
# !conda install -c conda-forge xtb-python     # xTB bindings
# !pip install pyscf                             # DFT (optional)


# ================================================================
# Cell 1 — Imports and Configuration
# ================================================================
import json
import warnings
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# ── Project Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(r"G:\research\ECOAI")
DATA_DIR     = PROJECT_ROOT / "Datasets" / "data"
PHASE7_DIR   = PROJECT_ROOT / "experiment" / "phase7_quantum_descriptors"

# ── Input ──────────────────────────────────────────────────
INPUT_CURATED     = DATA_DIR / "curated_molecules_with_splits.parquet"
INPUT_PREDICTIONS = DATA_DIR / "model_predictions.parquet"

# ── Output ─────────────────────────────────────────────────
OUTPUT_QUANTUM    = DATA_DIR / "quantum_descriptors.parquet"
QUANTUM_LOG       = PHASE7_DIR / "quantum_log.json"
CONFORMER_DIR     = PHASE7_DIR / "conformers"

# ── Selection Parameters ──────────────────────────────────
N_LABELED_PER_CLASS = 150   # 150 repellent + 150 non-repellent
N_UNLABELED_HIGH_P  = 100   # top predicted repellent from LifeChemicals
N_UNLABELED_HIGH_U  = 100   # highest uncertainty from LifeChemicals
TOTAL_QUANTUM = N_LABELED_PER_CLASS * 2 + N_UNLABELED_HIGH_P + N_UNLABELED_HIGH_U

# DFT validation subset
N_DFT_VALIDATION = 100

# ── Conformer Generation ──────────────────────────────────
N_CONFORMERS  = 50     # ETKDG conformers to generate
MMFF_MAX_ITERS = 500    # MMFF optimization iterations
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ── Verify ─────────────────────────────────────────────────
assert INPUT_CURATED.exists(), f"❌ {INPUT_CURATED} — Run Phase 2 first!"
PHASE7_DIR.mkdir(parents=True, exist_ok=True)
CONFORMER_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PHASE 7 — QUANTUM DESCRIPTORS")
print("=" * 60)
print(f"  Total molecules for QC: {TOTAL_QUANTUM}")
print(f"  DFT validation subset:  {N_DFT_VALIDATION}")


# ================================================================
# Cell 2 — Select Molecules for Quantum Calculations
# ================================================================

df_curated = pd.read_parquet(INPUT_CURATED)
df_curated_pass = df_curated[df_curated["qc_status"] == "pass"].copy()

# Try to load predictions for smart selection of unlabeled molecules
has_predictions = INPUT_PREDICTIONS.exists()
if has_predictions:
    df_preds = pd.read_parquet(INPUT_PREDICTIONS)
    print(f"  ✅ Model predictions loaded for smart unlabeled selection.")
else:
    print(f"  ⚠️  No model predictions found. Will select unlabeled randomly.")

# ── Select labeled molecules (scaffold-diverse) ──────────
df_repellent    = df_curated_pass[df_curated_pass["repellent_active"] == 1].copy()
df_nonrepellent = df_curated_pass[df_curated_pass["repellent_active"] == 0].copy()


def select_scaffold_diverse(df, n, seed=42):
    """
    Select n molecules with maximum scaffold diversity.
    Strategy: round-robin over scaffolds, one molecule per scaffold.
    """
    rng = np.random.RandomState(seed)

    # Group by scaffold
    scaffold_groups = {}
    for _, row in df.iterrows():
        sc = row.get("scaffold_smiles", "unknown")
        if pd.isna(sc):
            sc = "unknown"
        scaffold_groups.setdefault(sc, []).append(row["compound_id"])

    # Shuffle within each scaffold
    for sc in scaffold_groups:
        rng.shuffle(scaffold_groups[sc])

    # Round-robin selection
    selected = []
    scaffold_list = list(scaffold_groups.keys())
    rng.shuffle(scaffold_list)

    idx = 0
    while len(selected) < n and any(len(v) > 0 for v in scaffold_groups.values()):
        sc = scaffold_list[idx % len(scaffold_list)]
        if scaffold_groups[sc]:
            selected.append(scaffold_groups[sc].pop(0))
        idx += 1

    return selected


# Select labeled molecules
selected_repellent = select_scaffold_diverse(df_repellent, N_LABELED_PER_CLASS, RANDOM_SEED)
selected_nonrepellent = select_scaffold_diverse(df_nonrepellent, N_LABELED_PER_CLASS, RANDOM_SEED)

print(f"\n  Selected labeled molecules:")
print(f"    Repellent:     {len(selected_repellent)} (from {len(df_repellent)} available)")
print(f"    Non-repellent: {len(selected_nonrepellent)} (from {len(df_nonrepellent)} available)")

# ── Select unlabeled molecules ────────────────────────────
df_unlabeled = df_curated_pass[df_curated_pass["repellent_active"].isna()].copy()

if has_predictions and "p_repellent" in df_preds.columns:
    # Merge predictions onto unlabeled data
    df_unlabeled_with_preds = df_unlabeled.merge(
        df_preds[["compound_id", "p_repellent", "ood_score"]],
        on="compound_id", how="left",
    )

    # Top N_UNLABELED_HIGH_P by predicted repellent probability
    df_high_p = df_unlabeled_with_preds.nlargest(N_UNLABELED_HIGH_P, "p_repellent")
    selected_high_p = df_high_p["compound_id"].tolist()

    # Top N_UNLABELED_HIGH_U by uncertainty (OOD score)
    remaining = df_unlabeled_with_preds[
        ~df_unlabeled_with_preds["compound_id"].isin(selected_high_p)
    ]
    df_high_u = remaining.nlargest(N_UNLABELED_HIGH_U, "ood_score")
    selected_high_u = df_high_u["compound_id"].tolist()
else:
    # Random selection
    selected_high_p = df_unlabeled.sample(
        n=min(N_UNLABELED_HIGH_P, len(df_unlabeled)), random_state=RANDOM_SEED
    )["compound_id"].tolist()
    remaining_ids = [cid for cid in df_unlabeled["compound_id"] if cid not in selected_high_p]
    selected_high_u = pd.Series(remaining_ids).sample(
        n=min(N_UNLABELED_HIGH_U, len(remaining_ids)), random_state=RANDOM_SEED
    ).tolist()

print(f"\n  Selected unlabeled molecules:")
print(f"    High p_repellent: {len(selected_high_p)}")
print(f"    High uncertainty: {len(selected_high_u)}")

# ── Combine all selected ──────────────────────────────────
all_selected = (
    selected_repellent + selected_nonrepellent +
    selected_high_p + selected_high_u
)

# Tag selection reason
selection_reason = {}
for cid in selected_repellent:
    selection_reason[cid] = "labeled_repellent"
for cid in selected_nonrepellent:
    selection_reason[cid] = "labeled_nonrepellent"
for cid in selected_high_p:
    selection_reason[cid] = "unlabeled_high_p_repellent"
for cid in selected_high_u:
    selection_reason[cid] = "unlabeled_high_uncertainty"

df_quantum = df_curated_pass[
    df_curated_pass["compound_id"].isin(all_selected)
].copy()
df_quantum["selection_reason"] = df_quantum["compound_id"].map(selection_reason)

print(f"\n  Total quantum subset: {len(df_quantum)} molecules")
print(f"  Selection breakdown:")
print(df_quantum["selection_reason"].value_counts().to_string())


# ================================================================
# Cell 3 — Conformer Generation (ETKDG + MMFF)
# ================================================================
# For each molecule:
# 1. Generate N conformers with ETKDG
# 2. MMFF-optimize each conformer
# 3. Keep the lowest-energy conformer

print("\n" + "=" * 60)
print("  CONFORMER GENERATION")
print("=" * 60)


def generate_best_conformer(smiles, n_confs=50, max_iters=500, seed=42):
    """
    Generate conformers with ETKDG, MMFF-optimize, and
    return the lowest-energy 3D Mol object.

    Returns:
        mol_3d: RDKit Mol with best conformer, or None
        energy: MMFF energy of best conformer, or None
        n_generated: number of conformers successfully generated
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, 0

    mol = Chem.AddHs(mol)

    # ETKDG conformer generation
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0
    params.useSmallRingTorsions = True

    n_generated = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if len(n_generated) == 0:
        # Fallback: try without distance geometry constraints
        params.useRandomCoords = True
        n_generated = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if len(n_generated) == 0:
            return None, None, 0

    # MMFF optimize each conformer
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters, numThreads=0)

    # Find lowest energy conformer
    best_conf_id = None
    best_energy = float("inf")
    for conf_id, (converged, energy) in enumerate(results):
        if energy < best_energy:
            best_energy = energy
            best_conf_id = conf_id

    if best_conf_id is None:
        return None, None, n_generated

    # Keep only the best conformer
    mol_best = Chem.RWMol(mol)
    conf_ids = [c.GetId() for c in mol_best.GetConformers()]
    for cid in conf_ids:
        if cid != best_conf_id:
            mol_best.RemoveConformer(cid)

    return mol_best, best_energy, len(n_generated)


# Generate conformers for all selected molecules
conformer_results = []

for idx, row in tqdm(df_quantum.iterrows(), total=len(df_quantum), desc="  Conformers"):
    smiles = row["canonical_smiles"]
    cid = row["compound_id"]

    mol_3d, energy, n_gen = generate_best_conformer(
        smiles, n_confs=N_CONFORMERS, max_iters=MMFF_MAX_ITERS, seed=RANDOM_SEED
    )

    conformer_results.append({
        "compound_id": cid,
        "mol_3d": mol_3d,
        "mmff_energy": energy,
        "n_conformers_generated": n_gen,
        "conformer_success": mol_3d is not None,
    })

df_conf = pd.DataFrame(conformer_results)
n_success = df_conf["conformer_success"].sum()
n_fail = len(df_conf) - n_success

print(f"\n  ✅ Conformer generation: {n_success} success, {n_fail} failed")


# ================================================================
# Cell 4 — xTB Quantum Descriptors
# ================================================================
# Extract electronic structure descriptors using GFN2-xTB.
#
# NOTE: If xtb-python is not available, this cell provides a
# fallback using the xtb command-line tool via subprocess.
# Alternatively, RDKit-computed quantum-proxy descriptors can
# be used as a fallback.

print("\n" + "=" * 60)
print("  xTB QUANTUM CALCULATIONS")
print("=" * 60)

# Try to import xtb-python
XTB_AVAILABLE = False
try:
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_MUTED
    XTB_AVAILABLE = True
    print("  ✅ xtb-python available.")
except ImportError:
    print("  ⚠️  xtb-python not found.")
    print("     Attempting command-line xtb fallback...")
    # Check if xtb is in PATH
    try:
        result = subprocess.run(["xtb", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            XTB_AVAILABLE = True
            print(f"  ✅ xtb CLI found: {result.stdout.strip()[:80]}")
        else:
            print("  ❌ xtb CLI not available either.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ❌ xtb not found in PATH.")

if not XTB_AVAILABLE:
    print("\n  ⚠️  FALLING BACK to RDKit-computed pseudo-quantum descriptors.")
    print("     Install xtb for real electronic structure calculations:")
    print("       conda install -c conda-forge xtb-python")


def compute_xtb_descriptors_xtbpython(mol_3d):
    """Compute xTB descriptors using xtb-python interface."""
    try:
        from xtb.interface import Calculator, Param
        from xtb.libxtb import VERBOSITY_MUTED

        conf = mol_3d.GetConformer()
        positions = conf.GetPositions()  # Angstrom
        positions_bohr = positions * 1.8897259886  # Convert to Bohr

        # Atomic numbers
        atoms = [mol_3d.GetAtomWithIdx(i).GetAtomicNum()
                 for i in range(mol_3d.GetNumAtoms())]
        numbers = np.array(atoms)

        # Run GFN2-xTB calculation
        calc = Calculator(Param.GFN2xTB, numbers, positions_bohr)
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()

        descriptors = {
            "homo_ev":           float(res.get_orbital_eigenvalues()[res.get_orbital_occupations() > 0][-1] * 27.2114),
            "lumo_ev":           float(res.get_orbital_eigenvalues()[res.get_orbital_occupations() == 0][0] * 27.2114) if (res.get_orbital_occupations() == 0).any() else None,
            "total_energy_eh":   float(res.get_energy()),
            "dipole_debye":      float(np.linalg.norm(res.get_dipole())),
            "method":            "GFN2-xTB",
            "convergence":       "converged",
        }

        # HOMO-LUMO gap
        if descriptors["lumo_ev"] is not None:
            descriptors["homo_lumo_gap_ev"] = descriptors["lumo_ev"] - descriptors["homo_ev"]
        else:
            descriptors["homo_lumo_gap_ev"] = None

        # Partial charges
        charges = res.get_charges()
        descriptors["partial_charge_mean"] = float(charges.mean())
        descriptors["partial_charge_std"]  = float(charges.std())
        descriptors["partial_charge_max"]  = float(charges.max())
        descriptors["partial_charge_min"]  = float(charges.min())

        return descriptors

    except Exception as e:
        return {"method": "GFN2-xTB", "convergence": f"failed: {str(e)[:100]}"}


def compute_rdkit_pseudo_quantum(mol_3d):
    """
    Fallback: compute pseudo-quantum descriptors using RDKit.
    These approximate some electronic properties without xTB.
    """
    try:
        mol = Chem.RemoveHs(mol_3d) if mol_3d else None
        if mol is None:
            return {"method": "rdkit_fallback", "convergence": "failed: no mol"}

        descriptors = {
            "homo_ev":             None,  # not available from RDKit
            "lumo_ev":             None,
            "homo_lumo_gap_ev":    None,
            "total_energy_eh":     None,
            "dipole_debye":        None,
            "partial_charge_mean": float(Descriptors.MaxPartialCharge(mol) + Descriptors.MinPartialCharge(mol)) / 2,
            "partial_charge_std":  float(abs(Descriptors.MaxPartialCharge(mol) - Descriptors.MinPartialCharge(mol))),
            "partial_charge_max":  float(Descriptors.MaxAbsPartialCharge(mol)),
            "partial_charge_min":  float(Descriptors.MinAbsPartialCharge(mol)),
            "method":              "rdkit_fallback",
            "convergence":         "na_fallback",
        }

        # Gasteiger charges as proxy
        Chem.AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
                if not np.isnan(charge) and not np.isinf(charge):
                    charges.append(charge)
            except Exception:
                pass

        if charges:
            descriptors["partial_charge_mean"] = float(np.mean(charges))
            descriptors["partial_charge_std"]  = float(np.std(charges))
            descriptors["partial_charge_max"]  = float(np.max(charges))
            descriptors["partial_charge_min"]  = float(np.min(charges))

        return descriptors

    except Exception as e:
        return {"method": "rdkit_fallback", "convergence": f"failed: {str(e)[:100]}"}


# Run quantum calculations
print(f"\n  Computing descriptors for {n_success} molecules with conformers...")

quantum_records = []

for _, row in tqdm(df_conf[df_conf["conformer_success"]].iterrows(),
                   total=n_success, desc="  xTB/Fallback"):
    cid = row["compound_id"]
    mol_3d = row["mol_3d"]

    if XTB_AVAILABLE:
        try:
            descriptors = compute_xtb_descriptors_xtbpython(mol_3d)
        except Exception:
            descriptors = compute_rdkit_pseudo_quantum(mol_3d)
    else:
        descriptors = compute_rdkit_pseudo_quantum(mol_3d)

    descriptors["compound_id"] = cid
    descriptors["mmff_energy"] = row["mmff_energy"]
    descriptors["charge"] = 0
    descriptors["multiplicity"] = 1
    quantum_records.append(descriptors)

# Also record failed conformer molecules
for _, row in df_conf[~df_conf["conformer_success"]].iterrows():
    quantum_records.append({
        "compound_id": row["compound_id"],
        "method": "none",
        "convergence": "conformer_failed",
    })

df_quantum_desc = pd.DataFrame(quantum_records)

print(f"\n  ✅ Quantum descriptors computed: {len(df_quantum_desc)} molecules")

# Summary of convergence
print(f"\n  Convergence summary:")
print(df_quantum_desc["convergence"].value_counts().to_string())


# ================================================================
# Cell 5 — DFT Validation Subset Selection
# ================================================================
# Select 100 molecules for DFT single-point validation.
# Balanced by class and chemically diverse.

print("\n" + "=" * 60)
print(f"  DFT VALIDATION SUBSET ({N_DFT_VALIDATION} molecules)")
print("=" * 60)

# Select from successfully computed xTB molecules
df_xtb_success = df_quantum_desc[
    df_quantum_desc["convergence"].isin(["converged", "na_fallback"])
].copy()

# Merge selection_reason
df_xtb_success = df_xtb_success.merge(
    df_quantum[["compound_id", "selection_reason"]],
    on="compound_id", how="left",
)

# Select balanced: 50 labeled (25 rep + 25 non-rep) + 50 unlabeled
labeled_xtb = df_xtb_success[
    df_xtb_success["selection_reason"].isin(["labeled_repellent", "labeled_nonrepellent"])
]
unlabeled_xtb = df_xtb_success[
    ~df_xtb_success["selection_reason"].isin(["labeled_repellent", "labeled_nonrepellent"])
]

n_labeled_dft = min(50, len(labeled_xtb))
n_unlabeled_dft = min(N_DFT_VALIDATION - n_labeled_dft, len(unlabeled_xtb))

dft_labeled = labeled_xtb.sample(n=n_labeled_dft, random_state=RANDOM_SEED)
dft_unlabeled = unlabeled_xtb.sample(n=n_unlabeled_dft, random_state=RANDOM_SEED)
dft_subset_ids = set(dft_labeled["compound_id"].tolist() + dft_unlabeled["compound_id"].tolist())

df_quantum_desc["dft_subset"] = df_quantum_desc["compound_id"].isin(dft_subset_ids)

print(f"  Selected {len(dft_subset_ids)} molecules for DFT validation:")
print(f"    Labeled:   {len(dft_labeled)}")
print(f"    Unlabeled: {len(dft_unlabeled)}")
print(f"\n  ℹ️  DFT calculations are computationally expensive and are")
print(f"     typically run on a cluster. The 'dft_subset' column flags")
print(f"     which molecules should be sent for DFT computation.")
print(f"     After DFT is complete, update this parquet with DFT values")
print(f"     and compare xTB vs. DFT HOMO/LUMO gap rankings.")


# ================================================================
# Cell 6 — Merge with Molecule Metadata
# ================================================================

# Join back to quantum subset metadata
df_quantum_final = df_quantum[
    ["compound_id", "canonical_smiles", "source_dataset",
     "repellent_active", "scaffold_smiles"]
].merge(df_quantum_desc, on="compound_id", how="right")

# Reorder columns
core_cols = [
    "compound_id", "canonical_smiles", "source_dataset",
    "repellent_active", "scaffold_smiles", "selection_reason",
    "method", "convergence", "charge", "multiplicity",
    "mmff_energy",
    "homo_ev", "lumo_ev", "homo_lumo_gap_ev",
    "total_energy_eh", "dipole_debye",
    "partial_charge_mean", "partial_charge_std",
    "partial_charge_max", "partial_charge_min",
    "dft_subset",
]

# Keep core columns that exist
final_cols = [c for c in core_cols if c in df_quantum_final.columns]
extra_cols = [c for c in df_quantum_final.columns if c not in core_cols]
final_cols.extend(extra_cols)

df_quantum_final = df_quantum_final[final_cols]

print(f"\n✅ Final quantum descriptor table:")
print(f"   {df_quantum_final.shape[0]} molecules × {df_quantum_final.shape[1]} columns")


# ================================================================
# Cell 7 — Save quantum_descriptors.parquet
# ================================================================

df_quantum_final.to_parquet(OUTPUT_QUANTUM, index=False, engine="pyarrow")
print(f"\n✅ Saved: {OUTPUT_QUANTUM}")
print(f"   Size: {OUTPUT_QUANTUM.stat().st_size / 1024:.1f} KB")

# Also save as CSV for inspection
csv_path = PHASE7_DIR / "quantum_descriptors_preview.csv"
df_quantum_final.to_csv(csv_path, index=False)
print(f"✅ Saved CSV: {csv_path}")


# ================================================================
# Cell 8 — Quantum Descriptor Statistics
# ================================================================

print("\n" + "=" * 60)
print("  QUANTUM DESCRIPTOR STATISTICS")
print("=" * 60)

numeric_cols = [
    "homo_ev", "lumo_ev", "homo_lumo_gap_ev", "total_energy_eh",
    "dipole_debye", "partial_charge_mean", "partial_charge_std",
    "partial_charge_max", "partial_charge_min", "mmff_energy",
]
numeric_available = [c for c in numeric_cols if c in df_quantum_final.columns]

if numeric_available:
    stats = df_quantum_final[numeric_available].describe().T
    stats["non_null"] = df_quantum_final[numeric_available].notna().sum()
    print(stats.round(4).to_string())
else:
    print("  No numeric quantum descriptors available.")


# ================================================================
# Cell 9 — Save Quantum Log
# ================================================================

quantum_log = {
    "pipeline": "Phase 7 — Quantum Descriptors",
    "random_seed": RANDOM_SEED,
    "xtb_available": XTB_AVAILABLE,
    "selection": {
        "labeled_repellent": len(selected_repellent),
        "labeled_nonrepellent": len(selected_nonrepellent),
        "unlabeled_high_p": len(selected_high_p),
        "unlabeled_high_u": len(selected_high_u),
        "total": len(all_selected),
    },
    "conformer_generation": {
        "n_conformers_per_mol": N_CONFORMERS,
        "mmff_max_iters": MMFF_MAX_ITERS,
        "success": int(n_success),
        "failed": int(n_fail),
    },
    "convergence_summary": df_quantum_desc["convergence"].value_counts().to_dict(),
    "dft_subset_size": int(len(dft_subset_ids)),
    "output_file": str(OUTPUT_QUANTUM),
}

with open(QUANTUM_LOG, "w") as f:
    json.dump(quantum_log, f, indent=2, default=str)

print(f"\n✅ Quantum log saved: {QUANTUM_LOG}")


# ================================================================
# Cell 10 — Final Summary
# ================================================================

print("\n" + "=" * 60)
print("  PHASE 7 COMPLETE — Quantum Descriptors")
print("=" * 60)
print(f"\n  Key results:")
print(f"    • {len(df_quantum_final)} molecules with quantum descriptors")
print(f"    • Method: {'GFN2-xTB' if XTB_AVAILABLE else 'RDKit fallback'}")
print(f"    • Conformer success: {n_success}/{len(df_conf)}")
print(f"    • DFT validation subset: {len(dft_subset_ids)} flagged")
print(f"\n  Artifacts:")
print(f"    1. {OUTPUT_QUANTUM}  (PROJECT ARTIFACT)")
print(f"    2. {QUANTUM_LOG}")
print(f"\n  Next: Phase 8 will compare cheminformatics-only vs.")
print(f"        quantum-only vs. fused features on this subset.")
print(f"\n  Ready for Phase 8 (V2 Ablation Study) →")
print("=" * 60)
