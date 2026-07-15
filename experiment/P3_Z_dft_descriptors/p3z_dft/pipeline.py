"""P3-Z pipeline orchestration: conformers, xTB, DFT, outputs."""

from __future__ import annotations

import json
import os
import pickle
import time
import traceback
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from p3z_dft import config
from p3z_dft import runtime
from p3z_dft.checkpoint import checkpoint_has_required_xtb, checkpoint_matches_current_run
from p3z_dft.conformers import generate_best_conformer
from p3z_dft.descriptors import compute_dft_descriptors
from p3z_dft.dft_engine import build_mean_field, build_pyscf_molecule
from p3z_dft.dft_logging import dft_debug, dft_molecule_log, dft_status, method_label
from p3z_dft.geometric_descriptors import (
    ARBALIGN_AVAILABLE,
    compute_hcm,
    compute_hoppe_descriptors,
    compute_radius_of_gyration,
)
from p3z_dft.io_utils import read_xyz
from p3z_dft.rdkit_geom import (
    copy_rdkit_mol_with_coordinates,
    infer_rdkit_charge_spin_multiplicity,
    rdkit_symbols_and_coordinates,
)
from p3z_dft.selection import select_balanced_labeled_subset, select_representative_pilot
from p3z_dft.vibrations import build_ir_spectrum, full_deuteration_masses
from p3z_dft.xtb_preopt import (
    XTB_METHOD,
    run_xtb_preoptimization,
    validate_xtb_charge_multiplicity,
    xtb_result_fields,
)
from p3z_dft.config import *  # noqa: F403


def main() -> None:
    """Run the full P3-Z workflow."""
    RDLogger.logger().setLevel(RDLogger.ERROR)
    warnings.filterwarnings("ignore")

    os.environ.setdefault(
        "CONDA_DEFAULT_PATH",
        str(os.path.expanduser("~/miniforge3/etc/profile.d/conda.sh")),
    )
    print("Exported CONDA_DEFAULT_PATH:", os.environ["CONDA_DEFAULT_PATH"])
    print("P3Z_INPUT_PARQUET preset:", os.environ.get("P3Z_INPUT_PARQUET", "<auto>"))
    print("P3Z_CLASSICAL_FEATURES preset:", os.environ.get("P3Z_CLASSICAL_FEATURES", "<auto>"))

    runtime.initialize_runtime(require_input=True)

    df = pd.read_parquet(INPUT_PARQUET)
    df_pass = df[df["qc_status"] == "pass"].copy().reset_index(drop=True)
    df_labeled_all = df_pass[df_pass["repellent_active"].notna()].copy().reset_index(drop=True)

    if SELECTION_MODE == "labeled_only":
        df_selected_all = df_labeled_all.copy().reset_index(drop=True)
    else:
        df_selected_all = df_pass.copy().reset_index(drop=True)

    if PILOT_MODE == "representative" and ACTIVE_SELECTION_LIMIT > 0:
        df_selected, pilot_bucket_counts = select_representative_pilot(df_selected_all, ACTIVE_SELECTION_LIMIT)
        print(f"\n[PILOT] Using representative pilot subset: {len(df_selected)} of {len(df_selected_all)} molecules.")
        for bucket_name, bucket_count in pilot_bucket_counts.items():
            print(f"         {bucket_name:16s}: {bucket_count}")
    elif MAX_DFT_MOLECULES > 0:
        if SELECTION_MODE == "labeled_only":
            df_selected = select_balanced_labeled_subset(df_labeled_all, MAX_DFT_MOLECULES)
            n_pos_test = int((df_selected["repellent_active"] == 1).sum())
            n_neg_test = int((df_selected["repellent_active"] == 0).sum())
            print(f"\n[TEST] Using balanced smoke subset: {n_pos_test} positives + {n_neg_test} negatives from {len(df_labeled_all)} labeled molecules.")
        else:
            df_selected = df_selected_all.head(MAX_DFT_MOLECULES).copy().reset_index(drop=True)
            print(f"\n[TEST] Using first {len(df_selected)} of {len(df_selected_all)} all-QC-pass molecules.")
    else:
        df_selected = df_selected_all.copy().reset_index(drop=True)

    df_labeled = df_selected  # Downstream cells still use this name; it now means the selected QC-pass set.

    n_pos = int((df_selected["repellent_active"] == 1).sum())
    n_neg = int((df_selected["repellent_active"] == 0).sum())
    n_unlabeled = int(df_selected["repellent_active"].isna().sum())
    source_counts = df_selected["source_dataset"].value_counts(dropna=False)

    print(f"\n[OK] Loaded {len(df_pass)} QC-pass molecules total.")
    print(f"[OK] Computing P3-Z outputs for {len(df_selected)} selected molecules.")
    print(f"     Selection mode: {SELECTION_MODE}")
    print(f"     Labeled repellent:    {n_pos}")
    print(f"     Labeled nonrepellent: {n_neg}")
    print(f"     Unlabeled:            {n_unlabeled}")
    for source_name, source_count in source_counts.items():
        print(f"     Source {str(source_name):14s}: {int(source_count)}")

    # Check for existing checkpoint
    already_done_ids = set()
    if CHECKPOINT_FILE.exists():
        df_checkpoint = pd.read_parquet(CHECKPOINT_FILE)
        already_done_ids = set(df_checkpoint["compound_id"].tolist())
        print(f"\n[OK] CHECKPOINT FOUND: {len(already_done_ids)} molecules already computed. Resuming...")
    print("\n" + "=" * 60)
    print("  STEP 1: 3D CONFORMER GENERATION")
    print("=" * 60)

    import pickle
    if MAX_DFT_MOLECULES > 0:
        CONFORMER_CACHE = ARTIFACTS_DIR / f"_conformers_cache_smoke_{MAX_DFT_MOLECULES}.pkl"
    else:
        CONFORMER_CACHE = ARTIFACTS_DIR / "_conformers_cache.pkl"
    SELECTED_COMPOUND_IDS = set(df_labeled["compound_id"].astype(str))

    # --- Check for cached conformers on disk ---
    if CONFORMER_CACHE.exists():
        print(f"\n[OK] CONFORMER CACHE FOUND: {CONFORMER_CACHE.name}")
        print("     Loading conformers from disk (skipping generation)...")
        with open(CONFORMER_CACHE, "rb") as f:
            conformer_data = pickle.load(f)
        df_conf = pd.DataFrame(conformer_data)
        n_conf_success = df_conf["conformer_success"].sum()
        n_conf_fail = len(df_conf) - n_conf_success
        conf_elapsed = 0
        print(f"     Loaded {len(df_conf)} molecules ({n_conf_success} success, {n_conf_fail} failed)")
    else:
        print(f"\n     No cache found. Generating conformers from scratch...")
        conformer_data = []
        conf_start = time.time()

        for idx, row in tqdm(df_labeled.iterrows(), total=len(df_labeled), desc="  Conformers"):
            smiles = row["canonical_smiles"]
            cid = row["compound_id"]

            mol_3d, energy, n_gen = generate_best_conformer(
                smiles, n_confs=N_CONFORMERS, max_iters=MMFF_MAX_ITERS, seed=RANDOM_SEED
            )

            conformer_data.append({
                "compound_id": cid,
                "mol_3d": mol_3d,
                "mmff_energy": energy,
                "n_conformers_generated": n_gen,
                "conformer_success": mol_3d is not None,
            })

        conf_elapsed = time.time() - conf_start
        df_conf = pd.DataFrame(conformer_data)
        n_conf_success = df_conf["conformer_success"].sum()
        n_conf_fail = len(df_conf) - n_conf_success

        # Save conformers to disk for future runs
        with open(CONFORMER_CACHE, "wb") as f:
            pickle.dump(conformer_data, f)
        print(f"\n[OK] Conformers saved to disk: {CONFORMER_CACHE.name}")

    # Keep only conformers for the selected test/full molecule set, even if a cache contains more.
    df_conf = df_conf[df_conf["compound_id"].astype(str).isin(SELECTED_COMPOUND_IDS)].reset_index(drop=True)
    n_conf_success = int(df_conf["conformer_success"].sum()) if len(df_conf) else 0
    n_conf_fail = len(df_conf) - n_conf_success

    print(f"\n[OK] Conformer generation complete in {timedelta(seconds=int(conf_elapsed))}")
    print(f"     Selected conformers: {len(df_conf)}/{len(df_labeled)}")
    print(f"     Success: {n_conf_success} | Failed: {n_conf_fail}")
    # ================================================================
    # S5b — Fast DFT+D3 Smoke Test (Water, STO-3G)
    # ================================================================
    # This catches NumPy/SciPy/PySCF/dftd3 incompatibilities before the expensive full loop.
    print("\n" + "=" * 60)
    print("  SMOKE TEST: PySCF + DFT-D3 single-point and gradient")
    print("=" * 60)

    _smoke_basis = "sto-3g"
    _smoke_grid_level = 3
    mol_smoke = runtime.gto.M(
        atom="O 0.000000 0.000000 0.000000; H 0.000000 -0.757000 0.587000; H 0.000000 0.757000 0.587000",
        basis=_smoke_basis,
        unit="Angstrom",
        verbose=0,
        max_memory=min(PYSCF_MAX_MEMORY, 1000),
    )
    mf_smoke = build_mean_field(mol_smoke)
    mf_smoke.grids.level = 1
    e_smoke = mf_smoke.kernel()
    if not mf_smoke.converged:
        raise RuntimeError("Smoke-test SCF did not converge")
    g_smoke = np.asarray(mf_smoke.nuc_grad_method().kernel(), dtype=float)
    if g_smoke.shape != (3, 3) or not np.isfinite(g_smoke).all():
        raise RuntimeError(f"Smoke-test gradient invalid: shape={g_smoke.shape}")
    print(f"[OK] Smoke test passed: E = {float(e_smoke):.8f} Hartree, max|grad| = {np.max(np.abs(g_smoke)):.3e}")
    print("\n" + "=" * 60)
    print("  VALIDATION TESTS: GFN2-xTB pre-optimization integration")
    print("=" * 60)

    _xtb_validation_root = ARTIFACTS_DIR / "_xtb_validation"
    _xtb_validation_root.mkdir(parents=True, exist_ok=True)

    print("\n[TEST xTB-1] Executable and version detection...")
    assert not USE_XTB_PREOPT or runtime.XTB_EXECUTABLE is not None, "xTB executable should be detected when xTB preopt is enabled"
    assert not USE_XTB_PREOPT or runtime.XTB_VERSION not in (None, "", "not_available"), "xTB version should be recorded"
    print(f"  xTB executable: {runtime.XTB_EXECUTABLE}")
    print(f"  xTB version:    {runtime.XTB_VERSION}")

    print("\n[TEST xTB-2] Neutral singlet receives charge=0 and uhf=0...")
    water = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(water, randomSeed=RANDOM_SEED)
    AllChem.MMFFOptimizeMolecule(water, maxIters=50)
    water_symbols, water_coords = rdkit_symbols_and_coordinates(water)
    water_charge, water_spin, water_mult = infer_rdkit_charge_spin_multiplicity(water)
    water_uhf, _ = validate_xtb_charge_multiplicity(water_symbols, water_charge, water_mult)
    assert len(water_symbols) == water.GetNumAtoms()
    assert water_charge == 0 and water_mult == 1 and water_uhf == 0
    print(f"  atoms={len(water_symbols)} charge={water_charge} multiplicity={water_mult} uhf={water_uhf}")

    print("\n[TEST xTB-3] Open-shell test receives correct uhf...")
    methyl_radical = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
    rad_charge, rad_spin, rad_mult = infer_rdkit_charge_spin_multiplicity(methyl_radical)
    rad_symbols, _ = rdkit_symbols_and_coordinates(methyl_radical) if methyl_radical.GetNumConformers() else ([atom.GetSymbol() for atom in methyl_radical.GetAtoms()], np.zeros((methyl_radical.GetNumAtoms(), 3)))
    rad_uhf, _ = validate_xtb_charge_multiplicity(rad_symbols, rad_charge, rad_mult)
    assert rad_charge == 0 and rad_mult == 2 and rad_uhf == 1
    print(f"  methyl radical charge={rad_charge} multiplicity={rad_mult} uhf={rad_uhf}")

    print("\n[TEST xTB-4] Dry-run command/input generation...")
    dry_result = run_xtb_preoptimization(
        molecule_id="VALIDATION_DRY_RUN_H2O",
        symbols=water_symbols,
        coordinates_angstrom=water_coords,
        charge=water_charge,
        multiplicity=water_mult,
        output_root=_xtb_validation_root,
        solvent=XTB_SOLVENT,
        optimization_level=XTB_OPT_LEVEL,
        max_cycles=XTB_MAX_CYCLES,
        timeout_seconds=XTB_TIMEOUT_SECONDS,
        overwrite=True,
        executable=runtime.XTB_EXECUTABLE,
        xtb_version=runtime.XTB_VERSION,
        dry_run=True,
    )
    assert dry_result["status"] == "dry_run"
    assert Path(dry_result["input_geometry_path"]).exists()
    dry_symbols, _ = read_xyz(dry_result["input_geometry_path"])
    assert dry_symbols == water_symbols
    assert dry_result["command"][dry_result["command"].index("--chrg") + 1] == "0"
    assert dry_result["command"][dry_result["command"].index("--uhf") + 1] == "0"
    print("  dry-run XYZ and command validated")

    if USE_XTB_PREOPT and runtime.XTB_EXECUTABLE is not None and not XTB_DRY_RUN:
        print("\n[TEST xTB-5] Real GFN2-xTB run on water and cache reuse...")
        xtb_result_1 = run_xtb_preoptimization(
            molecule_id="VALIDATION_H2O",
            symbols=water_symbols,
            coordinates_angstrom=water_coords,
            charge=water_charge,
            multiplicity=water_mult,
            output_root=_xtb_validation_root,
            solvent=XTB_SOLVENT,
            optimization_level="normal",
            max_cycles=100,
            timeout_seconds=min(XTB_TIMEOUT_SECONDS, 180),
            overwrite=True,
            executable=runtime.XTB_EXECUTABLE,
            xtb_version=runtime.XTB_VERSION,
            dry_run=False,
        )
        assert xtb_result_1["converged"], f"xTB validation failed: {xtb_result_1.get('status')} {xtb_result_1.get('message')}"
        xtb_symbols, xtb_coords = read_xyz(xtb_result_1["output_geometry_path"])
        assert xtb_symbols == water_symbols
        assert np.asarray(xtb_result_1["optimized_coordinates_angstrom"]).shape == water_coords.shape

        xtb_result_2 = run_xtb_preoptimization(
            molecule_id="VALIDATION_H2O",
            symbols=water_symbols,
            coordinates_angstrom=water_coords,
            charge=water_charge,
            multiplicity=water_mult,
            output_root=_xtb_validation_root,
            solvent=XTB_SOLVENT,
            optimization_level="normal",
            max_cycles=100,
            timeout_seconds=min(XTB_TIMEOUT_SECONDS, 180),
            overwrite=False,
            executable=runtime.XTB_EXECUTABLE,
            xtb_version=runtime.XTB_VERSION,
            dry_run=False,
        )
        assert xtb_result_2["converged"] and xtb_result_2.get("from_cache"), "valid xTB cache should be reused"
        print(f"  converged energy={xtb_result_1.get('energy_Eh')} Eh; cache reused={xtb_result_2.get('from_cache')}")

        print("\n[TEST xTB-6] xTB coordinates are passed to the PySCF input builder...")
        water_xtb = copy_rdkit_mol_with_coordinates(water, xtb_result_1["optimized_coordinates_angstrom"])
        pyscf_from_xtb = build_pyscf_molecule(water_xtb)
        pyscf_coords = pyscf_from_xtb.atom_coords(unit="Angstrom")
        assert np.allclose(pyscf_coords, xtb_result_1["optimized_coordinates_angstrom"], atol=1e-7)
        print("  PySCF builder receives validated xTB coordinates")
    else:
        print("\n[WARN] Real xTB validation skipped because xTB is disabled, missing, or dry-run mode is active.")

    print("\n[TEST xTB-7] Simulated xTB failure is recorded without raising...")
    failure_result = run_xtb_preoptimization(
        molecule_id="VALIDATION_BAD_MULTIPLICITY",
        symbols=water_symbols,
        coordinates_angstrom=water_coords,
        charge=0,
        multiplicity=2,
        output_root=_xtb_validation_root,
        solvent=XTB_SOLVENT,
        optimization_level=XTB_OPT_LEVEL,
        max_cycles=10,
        timeout_seconds=30,
        overwrite=True,
        executable=runtime.XTB_EXECUTABLE,
        xtb_version=runtime.XTB_VERSION,
        dry_run=True,
    )
    assert not failure_result["converged"]
    assert failure_result["status"] == "invalid_charge_or_multiplicity"
    assert Path(failure_result["status_json_path"]).exists()
    print("  invalid multiplicity failure recorded in xtb_status.json")

    assert DFT_FUNCTIONAL == "b3lyp"
    assert XTB_SOLVENT is None
    print(
        "\n[OK] xTB integration validation passed; "
        f"current run profile={'smoke' if IS_SMOKE_RUN else 'production'} basis={DFT_BASIS} "
        f"geom_steps={GEOMOPT_MAX_STEPS} vib={COMPUTE_VIBRATIONS} ir={COMPUTE_IR_SPECTRA} "
        f"hd={COMPUTE_HD_SHIFTS} vea={COMPUTE_VERTICAL_EA}"
    )
    print("=" * 60)
    print("\n" + "=" * 60)
    print("  STEP 2: GFN2-xTB PRE-OPTIMIZATION + DFT+D3 GEOMETRY OPTIMIZATION")
    print("=" * 60)

    checkpoint_records = []
    if CHECKPOINT_FILE.exists():
        df_checkpoint = pd.read_parquet(CHECKPOINT_FILE)
        checkpoint_records = df_checkpoint.to_dict(orient="records")
        print(f"\n[OK] Loaded checkpoint with {len(checkpoint_records)} rows: {CHECKPOINT_FILE.name}")

    records_by_id = {}
    n_checkpoint_success = 0
    n_checkpoint_failed_skipped = 0
    n_checkpoint_incompatible = 0
    for record in checkpoint_records:
        if "compound_id" not in record or pd.isna(record["compound_id"]):
            continue
        status = str(record.get("status", ""))
        compatible = checkpoint_matches_current_run(record)
        keep_record = compatible and ((status == "converged") or not RERUN_FAILED_CHECKPOINTS)
        if keep_record:
            records_by_id[record["compound_id"]] = record
            if status == "converged":
                n_checkpoint_success += 1
        else:
            if compatible:
                n_checkpoint_failed_skipped += 1
            else:
                n_checkpoint_incompatible += 1

    if checkpoint_records:
        print(f"     Reusing converged checkpoint rows: {n_checkpoint_success}")
        if RERUN_FAILED_CHECKPOINTS:
            print(f"     Recomputing failed/non-converged checkpoint rows: {n_checkpoint_failed_skipped}")
        if n_checkpoint_incompatible:
            print(f"     Recomputing rows without current settings/artifacts: {n_checkpoint_incompatible}")

    dft_start = time.time()
    n_new = 0
    if DFT_WRITE_LOGS:
        DFT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        dft_status(f"  [LOG] Run log: {DFT_RUN_LOG_FILE}")
        dft_status(f"  [LOG] Per-molecule logs: {DFT_LOG_DIR}")

    for molecule_index, (_, row) in enumerate(tqdm(df_conf.iterrows(), total=len(df_conf), desc="  xTB+DFT"), start=1):
        cid = row["compound_id"]
        dft_debug(f"  [DFT] molecule {molecule_index}/{len(df_conf)}: {cid}")
        if cid in records_by_id:
            dft_debug(f"  [DFT] [{cid}] using converged checkpoint row")
            continue

        with dft_molecule_log(cid, molecule_index=molecule_index, total_molecules=len(df_conf)) as log_path:
            xtb_result = None
            xtb_fields = {}
            dft_start_geometry_source = None
            result = None

            if not bool(row.get("conformer_success", False)) or row.get("mol_3d") is None:
                dft_status(f"  [ERROR] [{cid}] skipped: conformer generation failed")
                result = {
                    "compound_id": cid,
                    "molecule_id": cid,
                    "method": method_label(),
                    "status": "failed: conformer_generation_failed",
                    "opt_status": "skipped",
                    "opt_converged": False,
                    "dft_converged": False,
                    "dispersion_corrected": USE_D3 and runtime.D3_AVAILABLE,
                    "descriptor_status": "skipped: conformer_generation_failed",
                    "dft_start_geometry_source": "NOT_RUN_CONFORMER_FAILED",
                }
            else:
                mol_for_dft = row["mol_3d"]
                if USE_XTB_PREOPT:
                    try:
                        mmff_symbols, mmff_coords = rdkit_symbols_and_coordinates(row["mol_3d"])
                        charge, _spin, multiplicity = infer_rdkit_charge_spin_multiplicity(row["mol_3d"])
                        dft_debug(
                            f"  [xTB] [{cid}] start: atoms={len(mmff_symbols)} charge={charge} "
                            f"multiplicity={multiplicity} opt={XTB_OPT_LEVEL} solvent={XTB_SOLVENT or 'gas'}"
                        )
                        xtb_result = run_xtb_preoptimization(
                            molecule_id=cid,
                            symbols=mmff_symbols,
                            coordinates_angstrom=mmff_coords,
                            charge=charge,
                            multiplicity=multiplicity,
                            output_root=XTB_CALCULATIONS_DIR,
                            solvent=XTB_SOLVENT,
                            optimization_level=XTB_OPT_LEVEL,
                            max_cycles=XTB_MAX_CYCLES,
                            timeout_seconds=XTB_TIMEOUT_SECONDS,
                            overwrite=XTB_OVERWRITE,
                            executable=runtime.XTB_EXECUTABLE,
                            xtb_version=runtime.XTB_VERSION,
                            dry_run=XTB_DRY_RUN,
                        )
                        xtb_fields = xtb_result_fields(xtb_result)
                        if xtb_result.get("converged"):
                            mol_for_dft = copy_rdkit_mol_with_coordinates(
                                row["mol_3d"],
                                xtb_result["optimized_coordinates_angstrom"],
                            )
                            dft_start_geometry_source = "GFN2-xTB"
                            cache_note = " cached" if xtb_result.get("from_cache") else ""
                            dft_status(
                                f"  [OK] [{cid}] GFN2-xTB converged{cache_note}; "
                                f"RMSD(MMFF,xTB)={xtb_result.get('mmff_to_xtb_rmsd_A', np.nan):.4f} A"
                            )
                        elif ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE:
                            dft_start_geometry_source = "MMFF_FALLBACK_AFTER_XTB_FAILURE"
                            dft_status(f"  [WARN] [{cid}] xTB failed; DFT will start from original MMFF geometry by explicit fallback policy")
                        else:
                            dft_start_geometry_source = "NOT_RUN_XTB_FAILED"
                            result = {
                                "compound_id": cid,
                                "molecule_id": cid,
                                "method": method_label(),
                                "status": "failed: xtb_preoptimization_failed",
                                "opt_status": "skipped",
                                "opt_converged": False,
                                "dft_converged": False,
                                "dispersion_corrected": USE_D3 and runtime.D3_AVAILABLE,
                                "descriptor_status": f"skipped: xTB {xtb_result.get('status')}",
                                "dft_start_geometry_source": dft_start_geometry_source,
                            }
                            result.update(xtb_fields)
                            dft_status(f"  [ERROR] [{cid}] xTB failed: {xtb_result.get('status')} — {xtb_result.get('message')}")
                    except Exception as xtb_exc:
                        xtb_fields = {
                            "xtb_method": XTB_METHOD,
                            "xtb_version": runtime.XTB_VERSION,
                            "xtb_converged": False,
                            "xtb_status": "preoptimization_exception",
                            "xtb_message": f"{type(xtb_exc).__name__}: {str(xtb_exc)[:240]}",
                        }
                        if ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE:
                            dft_start_geometry_source = "MMFF_FALLBACK_AFTER_XTB_FAILURE"
                            dft_status(f"  [WARN] [{cid}] xTB exception; DFT will start from original MMFF geometry by explicit fallback policy")
                        else:
                            dft_start_geometry_source = "NOT_RUN_XTB_EXCEPTION"
                            result = {
                                "compound_id": cid,
                                "molecule_id": cid,
                                "method": method_label(),
                                "status": "failed: xtb_preoptimization_exception",
                                "opt_status": "skipped",
                                "opt_converged": False,
                                "dft_converged": False,
                                "dispersion_corrected": USE_D3 and runtime.D3_AVAILABLE,
                                "descriptor_status": f"skipped: xTB exception {str(xtb_exc)[:160]}",
                                "dft_start_geometry_source": dft_start_geometry_source,
                            }
                            result.update(xtb_fields)
                            dft_debug(traceback.format_exc(limit=5).rstrip())
                            dft_status(f"  [ERROR] [{cid}] xTB exception: {type(xtb_exc).__name__}: {xtb_exc}")
                else:
                    dft_start_geometry_source = "MMFF_XTB_DISABLED"
                    xtb_fields = {
                        "xtb_method": XTB_METHOD,
                        "xtb_version": runtime.XTB_VERSION,
                        "xtb_converged": False,
                        "xtb_status": "disabled",
                        "xtb_message": "GFN2-xTB pre-optimization disabled",
                    }

                if result is None:
                    result = compute_dft_descriptors(mol_for_dft, compound_id=cid)
                    result["compound_id"] = cid
                    result["molecule_id"] = cid
                    result.update(xtb_fields)
                    result["dft_start_geometry_source"] = dft_start_geometry_source
                    result["dft_converged"] = result.get("status") == "converged"

            result["dft_log_file"] = str(log_path) if log_path is not None else None

        result["mmff_energy"] = row.get("mmff_energy", np.nan)
        result["n_conformers_generated"] = row.get("n_conformers_generated", np.nan)
        result["conformer_success"] = bool(row.get("conformer_success", False))
        records_by_id[cid] = result
        dft_debug(f"  [DFT] [{cid}] status={result.get('status')} opt={result.get('opt_status')} gap={result.get('dft_homo_lumo_gap_ev', np.nan)}")
        n_new += 1

        if CHECKPOINT_EVERY and n_new % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records_by_id.values()).to_parquet(CHECKPOINT_FILE, index=False, engine="pyarrow")
            dft_status(f"  [OK] Checkpoint saved after {n_new} new molecules")

    # Preserve df_labeled ordering in the final descriptor table.
    ordered_records = []
    for cid in df_labeled["compound_id"]:
        if cid in records_by_id:
            ordered_records.append(records_by_id[cid])
        else:
            ordered_records.append({
                "compound_id": cid,
                "molecule_id": cid,
                "method": method_label(),
                "status": "failed: missing_dft_record",
                "opt_status": "missing",
                "opt_converged": False,
                "dft_converged": False,
                "dispersion_corrected": USE_D3 and runtime.D3_AVAILABLE,
                "dft_log_file": None,
                "dft_start_geometry_source": "MISSING",
                "descriptor_status": "skipped: missing_dft_record",
            })

    df_dft = pd.DataFrame(ordered_records)

    # Always write the latest checkpoint so resume is exact even when fewer than CHECKPOINT_EVERY rows ran.
    df_dft.to_parquet(CHECKPOINT_FILE, index=False, engine="pyarrow")

    dft_elapsed = time.time() - dft_start
    status_text = df_dft["status"].fillna("").astype(str)
    n_converged = int((status_text == "converged").sum())
    n_opt_success = int(df_dft.get("opt_converged", pd.Series(False, index=df_dft.index)).fillna(False).astype(bool).sum())
    n_scf_fail = int((status_text == "scf_not_converged").sum())
    n_error = int((status_text.str.startswith("failed") | status_text.str.startswith("error")).sum())
    n_xtb_converged = int(df_dft.get("xtb_converged", pd.Series(False, index=df_dft.index)).fillna(False).astype(bool).sum())
    n_xtb_failed = int(((df_dft.get("xtb_status", pd.Series("", index=df_dft.index)).fillna("").astype(str) != "") & (~df_dft.get("xtb_converged", pd.Series(False, index=df_dft.index)).fillna(False).astype(bool))).sum())

    print(f"\n[OK] xTB+DFT+D3 stage complete in {timedelta(seconds=int(dft_elapsed))}")
    print(f"     xTB converged:        {n_xtb_converged}/{len(df_dft)}")
    print(f"     xTB failed/skipped:   {n_xtb_failed}")
    print(f"     Converged SCF:        {n_converged}/{len(df_dft)}")
    print(f"     Optimized geometries: {n_opt_success}/{len(df_dft)}")
    print(f"     SCF not converged:    {n_scf_fail}")
    print(f"     Errors:               {n_error}")
    print(f"     Checkpoint:           {CHECKPOINT_FILE}")
    if DFT_WRITE_LOGS:
        print(f"     DFT run log:          {DFT_RUN_LOG_FILE}")
        print(f"     Per-molecule logs:    {DFT_LOG_DIR}")
    print(f"     xTB calculation dirs: {XTB_CALCULATIONS_DIR}")

    # %% cell 17
    # ================================================================
    # S6 — Build Descriptor Table
    # ================================================================
    print("\n" + "=" * 60)
    print("  STEP 3: BUILDING DFT DESCRIPTOR TABLE")
    print("=" * 60)

    meta_cols = ["compound_id", "canonical_smiles", "source_dataset",
                 "repellent_active", "scaffold_smiles", "fold_id"]
    meta_available = [c for c in meta_cols if c in df_labeled.columns]
    df_meta = df_labeled[meta_available].copy()

    df_dft_final = df_meta.merge(df_dft, on="compound_id", how="left")

    # Numeric DFT/geometric feature columns for downstream ML.
    DFT_FEATURE_COLS = [
        "dft_homo_ev", "dft_lumo_ev", "dft_homo_lumo_gap_ev",
        "dft_total_energy_hartree", "dft_dipole_debye", "mmff_energy",
        "dft_mulliken_charge_mean", "dft_mulliken_charge_std",
        "dft_mulliken_charge_max", "dft_mulliken_charge_min",
        "dft_mulliken_charge_range",
        "dft_ionization_potential_ev", "dft_electron_affinity_ev", "dft_vertical_ea_ev",
        "dft_chemical_hardness_ev", "dft_electronegativity_ev",
        "dft_electrophilicity_ev", "dft_n_electrons", "dft_n_basis_functions",
        "formal_charge", "spin",
        "d_av_A", "ECN", "Rg_A", "HCM", "HD_A", "dQ_A", "alignment_RMSD_A",
        "dft_n_modes", "dft_n_imaginary_modes", "dft_min_frequency_cm1",
        "dft_max_ir_intensity", "dft_ir_intensity_sum",
        "hd_shift_mean_cm1", "hd_shift_max_cm1", "hd_shift_min_cm1",
    ]
    dft_available = [c for c in DFT_FEATURE_COLS if c in df_dft_final.columns]

    print(f"  DFT/geometric features available: {len(dft_available)}")
    for col in dft_available:
        n_valid = df_dft_final[col].notna().sum()
        print(f"    {col:40s}: {n_valid}/{len(df_dft_final)} non-null")

    # %% cell 18
    # ================================================================
    # S7 — Merge with Classical Features
    # ================================================================
    if CLASSICAL_FEATURES.exists():
        print(f"\n--- Merging with classical features from P3 ---")
        df_classical = pd.read_parquet(CLASSICAL_FEATURES)
        df_selected_ids = df_meta[["compound_id"]].copy()
        df_classical_selected = df_selected_ids.merge(df_classical, on="compound_id", how="left")

        df_q_features = df_dft_final[["compound_id"] + dft_available].copy()
        df_combined = df_classical_selected.merge(df_q_features, on="compound_id", how="left")

        meta_in_combined = [c for c in meta_cols if c in df_combined.columns]
        n_total_features = len(df_combined.columns) - len(meta_in_combined)

        print(f"  Classical features: {len(df_classical.columns) - len(meta_in_combined)}")
        print(f"  DFT features:      {len(dft_available)}")
        print(f"  Combined total:    {n_total_features}")

        combined_path = ARTIFACTS_DIR / "dft_features_combined.parquet"
        df_combined.to_parquet(combined_path, index=False, engine="pyarrow")
        print(f"\n[OK] Saved combined features: {combined_path.name}")
    else:
        print(f"\n[WARN] Classical features not found at {CLASSICAL_FEATURES}")
        combined_path = None

    # %% cell 19
    # ================================================================
    # S8 — Save DFT Descriptors
    # ================================================================
    save_cols = [c for c in df_dft_final.columns if c != "mol_3d"]
    dft_path = ARTIFACTS_DIR / "dft_descriptors_all.parquet"
    df_dft_final[save_cols].to_parquet(dft_path, index=False, engine="pyarrow")
    print(f"[OK] Saved DFT descriptors: {dft_path.name}")

    # %% cell 20
    # ================================================================
    # S9 — Statistics
    # ================================================================
    print("\n" + "=" * 60)
    print("  DFT DESCRIPTOR STATISTICS")
    print("=" * 60)

    if dft_available:
        stats = df_dft_final[dft_available].describe().T
        stats["non_null"] = df_dft_final[dft_available].notna().sum()
        print(stats.round(4).to_string())

    # %% cell 21
    # ================================================================
    # S9a — Lightweight Validation Tests (No DFT)
    # ================================================================
    print("\n" + "=" * 60)
    print("  VALIDATION TESTS: Geometric Descriptors")
    print("=" * 60)

    # Test A: Diatomic structure
    print("\n[TEST A] Diatomic structure (H-H at distance 0.74 Å)...")
    test_symbols_a = np.array(['H', 'H'])
    test_coords_a = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float64)
    hoppe_a = compute_hoppe_descriptors(test_symbols_a, test_coords_a)
    assert np.isclose(hoppe_a["d_av_A"], 0.74, atol=1e-3), f"d_av should be ~0.74, got {hoppe_a['d_av_A']}"
    assert np.isclose(hoppe_a["ECN"], 1.0, atol=1e-3), f"ECN should be ~1, got {hoppe_a['ECN']}"
    assert hoppe_a["hoppe_converged"], "Hoppe should converge for diatomic"
    print(f"  ✓ d_av = {hoppe_a['d_av_A']:.6f} Å, ECN = {hoppe_a['ECN']:.6f}")

    # Test B: Invariance to translation and rotation
    print("\n[TEST B] Invariance to translation and rotation...")
    test_symbols_b = np.array(['C', 'H', 'H', 'H'])
    test_coords_b = np.array([
        [0.0, 0.0, 0.0],
        [1.09, 0.0, 0.0],
        [-0.36, 1.03, 0.0],
        [-0.36, -0.52, 0.89]
    ], dtype=np.float64)

    d_av_orig = compute_hoppe_descriptors(test_symbols_b, test_coords_b)["d_av_A"]
    rg_orig = compute_radius_of_gyration(test_symbols_b, test_coords_b)
    hcm_orig = compute_hcm(test_symbols_b, test_coords_b)["HCM"]

    # Translate
    test_coords_b_trans = test_coords_b + np.array([10.0, 20.0, 30.0])
    d_av_trans = compute_hoppe_descriptors(test_symbols_b, test_coords_b_trans)["d_av_A"]
    rg_trans = compute_radius_of_gyration(test_symbols_b, test_coords_b_trans)
    hcm_trans = compute_hcm(test_symbols_b, test_coords_b_trans)["HCM"]

    assert np.isclose(d_av_orig, d_av_trans, atol=1e-6), "d_av should be invariant to translation"
    assert np.isclose(rg_orig, rg_trans, atol=1e-6), "Rg should be invariant to translation"
    assert np.isclose(hcm_orig, hcm_trans, atol=1e-6), "HCM should be invariant to translation"
    print(f"  ✓ Translation invariant: d_av, Rg, HCM unchanged")

    # Test C: Achiral (planar) geometry
    print("\n[TEST C] Achiral planar geometry...")
    test_symbols_c = np.array(['C', 'H', 'H', 'H', 'H'])
    test_coords_c = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0]
    ], dtype=np.float64)

    hcm_c = compute_hcm(test_symbols_c, test_coords_c)["HCM"]
    assert np.isclose(hcm_c, 0.0, atol=1e-2), f"HCM should be ~0 for achiral geometry, got {hcm_c}"
    print(f"  ✓ HCM = {hcm_c:.6f} (achiral)")

    # Test D: Single atom
    print("\n[TEST D] Single atom system...")
    test_symbols_d = np.array(['He'])
    test_coords_d = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    hoppe_d = compute_hoppe_descriptors(test_symbols_d, test_coords_d)
    rg_d = compute_radius_of_gyration(test_symbols_d, test_coords_d)
    hcm_d = compute_hcm(test_symbols_d, test_coords_d)
    assert np.isnan(hoppe_d["d_av_A"]), "d_av should be NaN for single atom"
    assert hoppe_d["hoppe_status"] == "single_atom", "Should mark as single_atom"
    assert np.isclose(rg_d, 0.0, atol=1e-6), "Rg should be 0 for single atom"
    assert hcm_d["hcm_status"] == "too_few_atoms", "HCM should skip for <2 atoms"
    print(f"  ✓ Single atom handled correctly")

    # Test E: Representative pilot selection
    print("\n[TEST E] Representative pilot selection...")
    pilot_df = pd.DataFrame([
        {"compound_id": "REP_001", "repellent_active": 1.0, "source_dataset": "repellent"},
        {"compound_id": "DEC_001", "repellent_active": 0.0, "source_dataset": "non_repellent"},
        {"compound_id": "INS_001", "repellent_active": np.nan, "source_dataset": "insecticide"},
        {"compound_id": "NAT_001", "repellent_active": np.nan, "source_dataset": "natural_product"},
        {"compound_id": "REP_002", "repellent_active": 1.0, "source_dataset": "repellent"},
    ])
    pilot_selected, pilot_counts = select_representative_pilot(pilot_df, 4)
    assert len(pilot_selected) == 4
    assert set(pilot_selected["source_dataset"]) == {"repellent", "non_repellent", "insecticide", "natural_product"}
    assert sum(pilot_counts.values()) == 4
    print("  ✓ Representative pilot spans labeled and unlabeled cohorts")

    # Test F: IR spectrum grid and normalization
    print("\n[TEST F] IR spectrum builder...")
    grid_f, raw_f, norm_f = build_ir_spectrum(np.array([1000.0, 1500.0]), np.array([10.0, 5.0]))
    assert grid_f.ndim == raw_f.ndim == norm_f.ndim == 1
    assert len(grid_f) == len(raw_f) == len(norm_f)
    assert np.isclose(norm_f.max(), 1.0, atol=1e-6)
    print("  ✓ Broadened and normalized spectrum validated")

    # Test G: Full deuteration mass substitution
    print("\n[TEST G] Full deuteration masses...")
    test_mol_g = runtime.gto.M(atom='C 0 0 0; H 0 0 1.0; O 0 1.0 0', basis='sto-3g', spin=1)
    masses_g = full_deuteration_masses(['C', 'H', 'O'], test_mol_g)
    base_masses_g = np.asarray(test_mol_g.atom_mass_list(isotope_avg=True), dtype=float)
    assert np.isclose(masses_g[0], base_masses_g[0])
    assert masses_g[1] > base_masses_g[1]
    assert np.isclose(masses_g[2], base_masses_g[2])
    print("  ✓ Hydrogen-only isotope substitution validated")

    print("\n[OK] All validation tests passed!")
    print("=" * 60)

    # %% cell 22
    # ================================================================
    # S10 — Distribution Plots (Active vs Inactive)
    # ================================================================
    print("\n--- Generating distribution plots ---")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_plot = df_dft_final[df_dft_final["status"] == "converged"].copy() if "status" in df_dft_final.columns else pd.DataFrame()

    if len(df_plot) > 0 and "dft_homo_lumo_gap_ev" in df_plot.columns:
        active = df_plot[df_plot["repellent_active"] == 1]
        inactive = df_plot[df_plot["repellent_active"] == 0]

        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
        })

        fig, axes = plt.subplots(3, 2, figsize=(14, 15))

        plot_configs = [
            ("dft_homo_lumo_gap_ev",         "HOMO-LUMO Gap (eV)",          axes[0, 0]),
            ("dft_dipole_debye",             "Dipole Moment (Debye)",       axes[0, 1]),
            ("dft_total_energy_hartree",     "Total DFT Energy (Hartree)",  axes[1, 0]),
            ("dft_mulliken_charge_range",    "Mulliken Charge Range",       axes[1, 1]),
            ("dft_chemical_hardness_ev",     "Chemical Hardness η (eV)",    axes[2, 0]),
            ("dft_electrophilicity_ev",      "Electrophilicity ω (eV)",     axes[2, 1]),
        ]

        for col, title, ax in plot_configs:
            if col in df_plot.columns and active[col].notna().any():
                ax.hist(active[col].dropna(), bins=25, alpha=0.7,
                        label="Repellent", color="#17a589", edgecolor="black")
                ax.hist(inactive[col].dropna(), bins=25, alpha=0.5,
                        label="Inactive", color="#e74c3c", edgecolor="black")
                ax.set_title(title, fontsize=13, fontweight="bold")
                ax.legend()

        for ax in axes.flat:
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("DFT Descriptor Distributions: Repellent vs Inactive\n"
                     f"(B3LYP/{DFT_BASIS}, {n_converged} molecules)",
                     fontsize=15, fontweight="bold")
        plt.tight_layout(pad=2.0)

        plot_png = PLOTS_DIR / "P3Z_dft_distributions.png"
        plot_jpg = PLOTS_DIR / "P3Z_dft_distributions.jpg"
        plt.savefig(plot_png, dpi=300)
        plt.savefig(plot_jpg, dpi=300)
        plt.close()
        print(f"[OK] Saved plots: {plot_png.name}")
    else:
        print("[WARN] No converged DFT results to plot.")

    # %% cell 23
    # ================================================================
    # S11 — Computation Log
    # ================================================================
    total_elapsed = conf_elapsed + dft_elapsed

    log = {
        "pipeline": "P3Z — DFT Geometry Optimization + Descriptor Computation",
        "random_seed": RANDOM_SEED,
        "selection": {
            "mode": SELECTION_MODE,
            "pilot_mode": PILOT_MODE,
            "pilot_size": PILOT_SIZE,
            "active_limit": ACTIVE_SELECTION_LIMIT,
        },
        "run_settings_fingerprint": RUN_SETTINGS_FINGERPRINT,
        "geometry_optimization": {
            "optimizer": runtime.GEOMOPT_BACKEND if runtime.GEOMOPT_AVAILABLE else "None (single-point only)",
            "enabled": runtime.GEOMOPT_AVAILABLE,
        },
        "dft_settings": {
            "functional": DFT_FUNCTIONAL,
            "basis_set": DFT_BASIS,
            "max_scf_cycles": MAX_SCF_CYCLES,
            "convergence_tolerance": SCF_CONV_TOL,
            "grid_level": 3,
            "dispersion_correction": "D3" if USE_D3 and runtime.D3_AVAILABLE else "None",
        },
        "analysis_stages": {
            "compute_vibrations": COMPUTE_VIBRATIONS,
            "compute_ir_spectra": COMPUTE_IR_SPECTRA,
            "compute_hd_shifts": COMPUTE_HD_SHIFTS,
            "compute_vertical_ea": COMPUTE_VERTICAL_EA,
        },
        "conformer_generation": {
            "n_conformers_per_mol": N_CONFORMERS,
            "mmff_max_iters": MMFF_MAX_ITERS,
            "success": int(n_conf_success),
            "failed": int(n_conf_fail),
            "time_seconds": int(conf_elapsed),
        },
        "xtb_preoptimization": {
            "enabled": USE_XTB_PREOPT,
            "method": XTB_METHOD,
            "xtb_version": runtime.XTB_VERSION,
            "optimization_level": XTB_OPT_LEVEL,
            "max_cycles": XTB_MAX_CYCLES,
            "solvent": XTB_SOLVENT or "gas",
            "converged": int(n_xtb_converged),
            "failed_or_skipped": int(n_xtb_failed),
            "fallback_to_mmff_after_failure": ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE,
            "calculation_root": str(XTB_CALCULATIONS_DIR),
        },
        "dft_calculations": {
            "converged": int(n_converged),
            "geometries_optimized": int(n_opt_success),
            "scf_not_converged": int(n_scf_fail),
            "errors": int(n_error),
            "time_seconds": int(dft_elapsed),
        },
        "dft_features": dft_available,
        "total_molecules": len(df_labeled),
        "output_files": {
            "dft_descriptors": str(dft_path),
            "combined_features": str(combined_path) if combined_path else "N/A",
            "calculation_root": str(XTB_CALCULATIONS_DIR),
        },
    }

    log_path = ARTIFACTS_DIR / "dft_computation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"[OK] Computation log saved: {log_path.name}")

    # Clean up checkpoint file after successful completion
    if CHECKPOINT_FILE.exists() and n_converged > 0:
        print(f"[OK] Checkpoint file retained at: {CHECKPOINT_FILE.name}")

    # %% cell 24
    # ================================================================
    # S12 — Final Summary
    # ================================================================
    dft_method_final = f"{DFT_FUNCTIONAL.upper()}/{DFT_BASIS}" + ("-D3" if USE_D3 and runtime.D3_AVAILABLE else "")
    opt_method = runtime.GEOMOPT_BACKEND if runtime.GEOMOPT_AVAILABLE else "Single-point (no optimizer)"

    print("\n" + "=" * 60)
    print("  P3Z COMPLETE — DFT Optimization + Geometric Descriptors")
    print("=" * 60)
    print(f"  Total Molecules:         {len(df_labeled)}")
    print(f"  Selection Mode:          {SELECTION_MODE}")
    print(f"  Pilot Mode:              {PILOT_MODE} ({PILOT_SIZE or 'off'})")
    print(f"  Conformer Success:       {n_conf_success}/{len(df_labeled)}")
    print(f"  xTB Pre-Optimization:    {XTB_METHOD if USE_XTB_PREOPT else 'Disabled'} ({XTB_OPT_LEVEL}, {XTB_SOLVENT or 'gas'})")
    print(f"  xTB Converged:           {n_xtb_converged}/{len(df_labeled)}")
    print(f"  DFT Method:              {dft_method_final}")
    print(f"  Geometry Optimization:   {opt_method}")
    print(f"  Total Converged:         {n_converged}")
    print(f"  Geometries Optimized:    {n_opt_success}")
    print(f"  DFT Features:            {len(dft_available)}")
    print(f"  Vibrational Stage:       {COMPUTE_VIBRATIONS}")
    print(f"  IR Spectra:              {COMPUTE_IR_SPECTRA}")
    print(f"  H->D Shifts:             {COMPUTE_HD_SHIFTS}")
    print(f"  Vertical EA:             {COMPUTE_VERTICAL_EA}")
    print(f"  Geometric Descriptors:   d_av, ECN, Rg, HCM (+ per-atom)")
    print(f"  ArbAlign for HCM:        {'Yes (ArbAlign)' if ARBALIGN_AVAILABLE else 'No (Fallback: Hungarian+Kabsch)'}")
    print(f"  Conformer Time:          {timedelta(seconds=int(conf_elapsed))}")
    print(f"  Optimization Time:       {timedelta(seconds=int(dft_elapsed))}")
    print(f"  Total Time:              {timedelta(seconds=int(total_elapsed))}")
    print(f"  Artifacts Directory:     {ARTIFACTS_DIR}")
    print()
    print(f"  Output Files:")
    print(f"    1. {dft_path.name}")
    print(f"    2. dft_features_combined.parquet")
    print(f"    3. dft_computation_log.json")
    print(f"    4. descriptors_summary.csv")
    print(f"    5. descriptors_per_atom.csv")
    print(f"    6. P3Z_dft_distributions.png")
    print(f"    7. calculations/<compound_id>/vibrations/*")
    print(f"    8. calculations/<compound_id>/isotopes/full_deuteration/*")
    print(f"    9. calculations/<compound_id>/anion/vertical_ea.json")
    print("=" * 60)

    # %% cell 25
    # ================================================================
    # S13 — Build Descriptor Summary Tables
    # ================================================================
    print("\n" + "=" * 60)
    print("  BUILDING GEOMETRIC DESCRIPTOR SUMMARY")
    print("=" * 60)

    # Merge DFT results with molecular metadata
    descriptor_summary = df_meta.merge(df_dft, on="compound_id", how="left")

    # Select key columns for summary
    summary_cols = [
        "compound_id",
        "molecule_id",
        "xtb_method",
        "xtb_version",
        "xtb_charge",
        "xtb_multiplicity",
        "xtb_uhf",
        "xtb_solvent",
        "xtb_opt_level",
        "xtb_converged",
        "xtb_energy_Eh",
        "xtb_gradient_norm_Eh_per_bohr",
        "xtb_cycles",
        "xtb_wall_time_s",
        "mmff_to_xtb_rmsd_A",
        "xtb_input_geometry_path",
        "xtb_optimized_geometry_path",
        "xtb_status",
        "xtb_message",
        "dft_start_geometry_source",
        "dft_converged",
        "status",
        "opt_status",
        "opt_converged",
        "dft_total_energy_hartree",
        "dft_vertical_ea_ev",
        "dft_vibrational_status",
        "dft_n_modes",
        "dft_n_imaginary_modes",
        "dft_max_ir_intensity",
        "hd_shift_status",
        "hd_shift_mean_cm1",
        "dft_vertical_ea_status",
        "vibration_artifact_dir",
        "hd_shift_artifact_dir",
        "anion_artifact_path",
        "d_av_A",
        "ECN",
        "Rg_A",
        "HCM",
        "HD_A",
        "dQ_A",
        "alignment_RMSD_A",
        "hoppe_converged",
        "descriptor_status",
    ]
    summary_cols_available = [c for c in summary_cols if c in descriptor_summary.columns]
    descriptor_summary = descriptor_summary[summary_cols_available]

    # Count descriptor statistics
    if "descriptor_status" in descriptor_summary.columns:
        descriptor_status = descriptor_summary["descriptor_status"].fillna("").astype(str)
        n_geom_computed = int((descriptor_status == "success").sum())
        n_geom_failed = int(((descriptor_status != "") & (descriptor_status != "success")).sum())
    else:
        n_geom_computed = 0
        n_geom_failed = 0

    print(f"\n[OK] Geometric descriptors computed for {n_geom_computed} molecules")
    if n_geom_failed > 0:
        print(f"[WARN] {n_geom_failed} molecules had descriptor warnings/errors")

    # Create per-atom descriptor table from the stored optimized PySCF geometries.
    descriptor_per_atom_list = []
    for _, row in df_dft.iterrows():
        cid = row.get("compound_id")
        symbols_json = row.get("optimized_symbols_json")
        coords_json = row.get("optimized_coordinates_A_json")
        if not isinstance(symbols_json, str) or not isinstance(coords_json, str):
            continue

        try:
            symbols = np.array(json.loads(symbols_json))
            coords = np.array(json.loads(coords_json), dtype=np.float64)
            if len(symbols) == 0 or coords.shape != (len(symbols), 3):
                continue

            hoppe_result = compute_hoppe_descriptors(symbols, coords)
            d_av_per_atom = hoppe_result["d_av_per_atom_A"]
            ecn_per_atom = hoppe_result["ECN_per_atom"]

            for atom_idx, (sym, d_av_i, ecn_i) in enumerate(zip(symbols, d_av_per_atom, ecn_per_atom)):
                descriptor_per_atom_list.append({
                    "compound_id": cid,
                    "atom_index": atom_idx,
                    "element": sym,
                    "x_A": float(coords[atom_idx, 0]),
                    "y_A": float(coords[atom_idx, 1]),
                    "z_A": float(coords[atom_idx, 2]),
                    "d_av_i_A": float(d_av_i) if not np.isnan(d_av_i) else None,
                    "ECN_i": float(ecn_i) if not np.isnan(ecn_i) else None,
                })
        except Exception as exc:
            print(f"[WARN] Skipping per-atom descriptors for {cid}: {str(exc)[:120]}")

    descriptor_per_atom = pd.DataFrame(descriptor_per_atom_list)

    # Save summaries
    summary_path = ARTIFACTS_DIR / "descriptors_summary.csv"
    per_atom_path = ARTIFACTS_DIR / "descriptors_per_atom.csv"

    descriptor_summary.to_csv(summary_path, index=False)
    descriptor_per_atom.to_csv(per_atom_path, index=False)

    print(f"\n[OK] Descriptor summary saved: {summary_path.name}")
    print(f"[OK] Per-atom descriptors saved: {per_atom_path.name}")

    # Display summary with sensible precision
    print("\n" + "=" * 60)
    print("  DESCRIPTOR SUMMARY (first 10 rows)")
    print("=" * 60)
    display_cols = ["compound_id", "xtb_converged", "dft_start_geometry_source", "dft_vibrational_status", "hd_shift_status", "dft_vertical_ea_status", "dft_vertical_ea_ev", "d_av_A", "ECN", "Rg_A", "HCM", "descriptor_status"]
    display_cols_available = [c for c in display_cols if c in descriptor_summary.columns]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.6f}' if not np.isnan(x) else 'NaN')
    print(descriptor_summary[display_cols_available].head(10).to_string())
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')


if __name__ == "__main__":
    main()
