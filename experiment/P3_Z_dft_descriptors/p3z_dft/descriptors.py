from __future__ import annotations

import json
import time
import traceback

import numpy as np

from p3z_dft import config
from p3z_dft.config import (
    ALLOW_SINGLE_POINT_FALLBACK,
    COMPUTE_HD_SHIFTS,
    COMPUTE_VERTICAL_EA,
    COMPUTE_VIBRATIONS,
    RUN_SETTINGS_FINGERPRINT,
)
from p3z_dft.dft_engine import (
    build_pyscf_molecule,
    flatten_mo_energies_and_occ,
    optimize_geometry_scipy,
    run_final_single_point,
)
from p3z_dft.dft_logging import dft_debug, dft_status, method_label
from p3z_dft import runtime
from p3z_dft.geometric_descriptors import compute_hcm, compute_hoppe_descriptors, compute_radius_of_gyration
from p3z_dft.vibrations import (
    EV_PER_HARTREE,
    compute_full_deuteration_outputs,
    compute_vertical_ea_outputs,
    compute_vibrational_outputs,
)

def compute_dft_descriptors(mol_3d, compound_id=None):
    """
    Perform DFT+D3 geometry optimization and compute electronic + geometric descriptors.

    The final electronic descriptors and geometric descriptors are computed at the optimized
    geometry. Optional vibrational, isotope-shift, and anion single-point stages write their
    large outputs as sidecar artifacts under the per-compound calculation directory.
    """

    def stage_defaults(upstream_failure=False):
        return {
            "run_settings_fingerprint": RUN_SETTINGS_FINGERPRINT,
            "dft_vertical_ea_ev": np.nan,
            "dft_vertical_ea_status": "skipped_upstream_failure" if (upstream_failure and COMPUTE_VERTICAL_EA) else ("not_requested" if not COMPUTE_VERTICAL_EA else "pending"),
            "dft_vibrational_status": "skipped_upstream_failure" if (upstream_failure and COMPUTE_VIBRATIONS) else ("not_requested" if not COMPUTE_VIBRATIONS else "pending"),
            "dft_n_modes": np.nan,
            "dft_n_imaginary_modes": np.nan,
            "dft_min_frequency_cm1": np.nan,
            "dft_max_ir_intensity": np.nan,
            "dft_ir_intensity_sum": np.nan,
            "hd_shift_status": "skipped_upstream_failure" if (upstream_failure and COMPUTE_HD_SHIFTS) else ("not_requested" if not COMPUTE_HD_SHIFTS else "pending"),
            "hd_shift_mean_cm1": np.nan,
            "hd_shift_max_cm1": np.nan,
            "hd_shift_min_cm1": np.nan,
            "vibration_artifact_dir": None,
            "ir_spectrum_artifact_path": None,
            "hd_shift_artifact_dir": None,
            "anion_artifact_path": None,
        }

    try:
        initial_mol = build_pyscf_molecule(mol_3d)
        mol_for_final = initial_mol
        opt_status = "single_point"
        opt_converged = False
        opt_iterations = 0
        opt_evaluations = 0
        opt_max_gradient = np.nan

        dft_debug(f"  [DFT] [{compound_id or 'molecule'}] start")

        if runtime.GEOMOPT_AVAILABLE:
            try:
                mol_for_final, opt_info = optimize_geometry_scipy(initial_mol, compound_id=compound_id)
                opt_iterations = opt_info["n_iterations"]
                opt_evaluations = opt_info["n_evaluations"]
                opt_max_gradient = opt_info["max_gradient_hartree_per_bohr"]
                opt_converged = bool(opt_info["success"])
                if opt_converged:
                    opt_status = "optimized"
                    if compound_id:
                        dft_status(f"  [OK] [{compound_id}] Geometry optimization converged")
                else:
                    opt_status = f"opt_not_converged: {opt_info['message'][:140]}"
                    if compound_id:
                        dft_status(f"  [WARN] [{compound_id}] Optimization did not fully converge; using last geometry")
                    if not ALLOW_SINGLE_POINT_FALLBACK:
                        result = {
                            "method": method_label(),
                            "status": f"failed: {opt_status}",
                            "opt_status": opt_status,
                            "opt_converged": False,
                            "opt_iterations": opt_iterations,
                            "opt_evaluations": opt_evaluations,
                            "opt_max_gradient_hartree_per_bohr": opt_max_gradient,
                            "dispersion_corrected": config.USE_D3 and runtime.D3_AVAILABLE,
                            "descriptor_status": "skipped: optimization not converged",
                        }
                        result.update(stage_defaults(upstream_failure=True))
                        return result
            except Exception as opt_error:
                opt_status = f"opt_failed: {str(opt_error)[:160]}"
                mol_for_final = initial_mol
                dft_debug(traceback.format_exc(limit=4).rstrip())
                if compound_id:
                    dft_status(f"  [WARN] [{compound_id}] Optimization failed; using final single-point fallback")
                if not ALLOW_SINGLE_POINT_FALLBACK:
                    result = {
                        "method": method_label(),
                        "status": f"failed: {opt_status}",
                        "opt_status": opt_status,
                        "opt_converged": False,
                        "opt_iterations": opt_iterations,
                        "opt_evaluations": opt_evaluations,
                        "opt_max_gradient_hartree_per_bohr": opt_max_gradient,
                        "dispersion_corrected": config.USE_D3 and runtime.D3_AVAILABLE,
                        "descriptor_status": "skipped: optimization failed",
                    }
                    result.update(stage_defaults(upstream_failure=True))
                    return result

        dft_debug(f"  [DFT] [{compound_id or 'molecule'}] final single-point start")
        sp_start = time.time()
        mf, total_energy = run_final_single_point(mol_for_final)
        dft_debug(
            f"  [DFT] [{compound_id or 'molecule'}] final SCF done: "
            f"E={float(total_energy):.10f} Ha converged={getattr(mf, 'converged', False)} "
            f"time={time.time() - sp_start:.1f}s"
        )

        if not getattr(mf, "converged", False):
            result = {
                "method": method_label(),
                "status": "scf_not_converged",
                "opt_status": opt_status,
                "opt_converged": opt_converged,
                "opt_iterations": opt_iterations,
                "opt_evaluations": opt_evaluations,
                "opt_max_gradient_hartree_per_bohr": opt_max_gradient,
                "dispersion_corrected": config.USE_D3 and runtime.D3_AVAILABLE,
                "descriptor_status": "skipped: scf_not_converged",
            }
            result.update(stage_defaults(upstream_failure=True))
            return result

        # --- Extract Kohn-Sham orbital energies ---
        mo_energies_hartree, mo_occ = flatten_mo_energies_and_occ(mf)
        mo_energies_ev = mo_energies_hartree * EV_PER_HARTREE

        occupied = mo_energies_ev[mo_occ > 0]
        unoccupied = mo_energies_ev[mo_occ == 0]
        homo_ev = float(occupied[-1]) if len(occupied) else np.nan
        lumo_ev = float(unoccupied[0]) if len(unoccupied) else np.nan
        homo_lumo_gap = float(lumo_ev - homo_ev) if np.isfinite(homo_ev) and np.isfinite(lumo_ev) else np.nan

        # --- Dipole moment ---
        dipole_debye = float(np.linalg.norm(mf.dip_moment(unit="Debye", verbose=0)))

        # --- Mulliken Population Analysis ---
        mulliken_pop = mf.mulliken_pop(verbose=0)
        charges = np.asarray(mulliken_pop[1], dtype=float)  # Mulliken charges per atom

        charge_mean = float(np.mean(charges))
        charge_std = float(np.std(charges))
        charge_max = float(np.max(charges))
        charge_min = float(np.min(charges))
        charge_range = float(charge_max - charge_min)

        # --- Conceptual DFT Descriptors (Koopmans' theorem) ---
        ip = -homo_ev if np.isfinite(homo_ev) else np.nan
        ea = -lumo_ev if np.isfinite(lumo_ev) else np.nan

        if np.isfinite(ip) and np.isfinite(ea):
            eta = float((ip - ea) / 2.0)    # Chemical Hardness
            chi = float((ip + ea) / 2.0)    # Electronegativity
            omega = float((chi ** 2) / (2.0 * eta)) if eta != 0 else np.nan
        else:
            eta, chi, omega = np.nan, np.nan, np.nan

        # --- Geometric descriptors from optimized geometry ---
        geom_status = "success"
        d_av = np.nan
        ecn = np.nan
        rg = np.nan
        hcm = np.nan
        hd = np.nan
        dq = np.nan
        alignment_rmsd = np.nan
        hoppe_converged = False
        symbols = np.array([mol_for_final.atom_symbol(i) for i in range(mol_for_final.natm)], dtype=object)
        coords = np.asarray(mol_for_final.atom_coords(unit="Angstrom"), dtype=np.float64)

        try:
            hoppe_result = compute_hoppe_descriptors(symbols, coords)
            d_av = hoppe_result["d_av_A"]
            ecn = hoppe_result["ECN"]
            hoppe_converged = hoppe_result["hoppe_converged"]
            if not hoppe_converged:
                geom_status = f"hoppe_not_converged: {hoppe_result['hoppe_status']}"

            rg = compute_radius_of_gyration(symbols, coords)

            hcm_result = compute_hcm(symbols, coords)
            hcm = hcm_result["HCM"]
            hd = hcm_result["HD_A"]
            dq = hcm_result["dQ_A"]
            alignment_rmsd = hcm_result["alignment_RMSD_A"]
            if hcm_result["hcm_status"] != "success" and geom_status == "success":
                geom_status = hcm_result["hcm_status"]
        except Exception as exc:
            geom_status = f"descriptor_error: {str(exc)[:160]}"

        dft_debug(
            f"  [DFT] [{compound_id or 'molecule'}] descriptors: "
            f"HOMO={homo_ev:.4f} eV LUMO={lumo_ev:.4f} eV gap={homo_lumo_gap:.4f} eV "
            f"opt_status={opt_status}"
        )

        descriptors = {
            "dft_homo_ev": homo_ev,
            "dft_lumo_ev": lumo_ev,
            "dft_homo_lumo_gap_ev": homo_lumo_gap,
            "dft_total_energy_hartree": float(total_energy),
            "dft_dipole_debye": dipole_debye,
            "dft_mulliken_charge_mean": charge_mean,
            "dft_mulliken_charge_std": charge_std,
            "dft_mulliken_charge_max": charge_max,
            "dft_mulliken_charge_min": charge_min,
            "dft_mulliken_charge_range": charge_range,
            "dft_ionization_potential_ev": ip,
            "dft_electron_affinity_ev": ea,
            "dft_chemical_hardness_ev": eta,
            "dft_electronegativity_ev": chi,
            "dft_electrophilicity_ev": omega,
            "dft_n_electrons": int(mol_for_final.nelectron),
            "dft_n_basis_functions": int(mol_for_final.nao_nr()),
            "formal_charge": int(mol_for_final.charge),
            "spin": int(mol_for_final.spin),
            "method": method_label(),
            "status": "converged",
            "opt_status": opt_status,
            "opt_converged": opt_converged,
            "opt_iterations": opt_iterations,
            "opt_evaluations": opt_evaluations,
            "opt_max_gradient_hartree_per_bohr": opt_max_gradient,
            "dispersion_corrected": config.USE_D3 and runtime.D3_AVAILABLE,
            "dispersion_model": "DFT-D3" if config.USE_D3 and runtime.D3_AVAILABLE else "none",
            # Geometric descriptors
            "d_av_A": d_av,
            "ECN": ecn,
            "Rg_A": rg,
            "HCM": hcm,
            "HD_A": hd,
            "dQ_A": dq,
            "alignment_RMSD_A": alignment_rmsd,
            "hoppe_converged": hoppe_converged,
            "descriptor_status": geom_status,
            "optimized_symbols_json": json.dumps(symbols.tolist()),
            "optimized_coordinates_A_json": json.dumps(coords.tolist()),
        }
        descriptors.update(stage_defaults(upstream_failure=False))

        compound_key = str(compound_id or "molecule")
        if COMPUTE_VIBRATIONS:
            try:
                vib_fields, vib_stage = compute_vibrational_outputs(mol_for_final, compound_key)
                descriptors.update(vib_fields)
                dft_status(f"  [OK] [{compound_key}] Vibrational analysis complete ({vib_fields['dft_n_modes']} modes)")
                if COMPUTE_HD_SHIFTS:
                    hd_fields = compute_full_deuteration_outputs(compound_key, mol_for_final, vib_stage)
                    descriptors.update(hd_fields)
                    if hd_fields["hd_shift_status"] == "success":
                        dft_status(f"  [OK] [{compound_key}] H->D full-deuteration analysis complete")
            except Exception as vib_exc:
                descriptors.update({
                    "dft_vibrational_status": f"failed: {str(vib_exc)[:160]}",
                    "dft_n_modes": np.nan,
                    "dft_n_imaginary_modes": np.nan,
                    "dft_min_frequency_cm1": np.nan,
                    "dft_max_ir_intensity": np.nan,
                    "dft_ir_intensity_sum": np.nan,
                    "vibration_artifact_dir": None,
                    "ir_spectrum_artifact_path": None,
                })
                if COMPUTE_HD_SHIFTS:
                    descriptors.update({
                        "hd_shift_status": "skipped_vibration_failure",
                        "hd_shift_mean_cm1": np.nan,
                        "hd_shift_max_cm1": np.nan,
                        "hd_shift_min_cm1": np.nan,
                        "hd_shift_artifact_dir": None,
                    })
                dft_debug(traceback.format_exc(limit=5).rstrip())
                dft_status(f"  [WARN] [{compound_key}] Vibrational stage failed: {type(vib_exc).__name__}: {vib_exc}")

        if COMPUTE_VERTICAL_EA:
            try:
                descriptors.update(
                    compute_vertical_ea_outputs(
                        symbols=symbols.tolist(),
                        coords=coords,
                        neutral_charge=int(mol_for_final.charge),
                        neutral_spin=int(mol_for_final.spin),
                        compound_id=compound_key,
                    )
                )
                dft_status(f"  [OK] [{compound_key}] Vertical electron affinity computed")
            except Exception as ea_exc:
                descriptors.update({
                    "dft_vertical_ea_status": f"failed: {str(ea_exc)[:160]}",
                    "dft_vertical_ea_ev": np.nan,
                    "anion_artifact_path": None,
                })
                dft_debug(traceback.format_exc(limit=5).rstrip())
                dft_status(f"  [WARN] [{compound_key}] Vertical EA stage failed: {type(ea_exc).__name__}: {ea_exc}")

        return descriptors

    except Exception as exc:
        error_msg = str(exc)[:240]
        dft_debug(traceback.format_exc(limit=5).rstrip())
        if compound_id:
            dft_status(f"  [ERROR] [{compound_id}] FAILED: {error_msg}")
        result = {
            "method": method_label(),
            "status": f"failed: {error_msg}",
            "opt_status": "error",
            "opt_converged": False,
            "dispersion_corrected": config.USE_D3 and runtime.D3_AVAILABLE,
            "descriptor_status": f"skipped: {error_msg}",
        }
        result.update(stage_defaults(upstream_failure=True))
        return result

