from __future__ import annotations

import json

import numpy as np
from rdkit import Chem
from scipy.constants import physical_constants

from p3z_dft import runtime

from p3z_dft.config import (
    COMPUTE_HD_SHIFTS,
    COMPUTE_IR_SPECTRA,
    COMPUTE_VIBRATIONS,
    DFT_BASIS,
    DFT_FUNCTIONAL,
    IR_SPECTRUM_FWHM_CM1,
    IR_SPECTRUM_MAX_CM1,
    IR_SPECTRUM_MIN_CM1,
    IR_SPECTRUM_STEP_CM1,
    VIBRATION_GRID_LEVEL,
    VIBRATION_SCF_CONV_TOL,
    VIBRATION_SCF_MAX_CYCLES,
    ANION_GRID_LEVEL,
    ANION_SCF_CONV_TOL,
    ANION_SCF_MAX_CYCLES,
    USE_D3,
)
from p3z_dft.dft_engine import build_mean_field, build_pyscf_molecule_from_geometry, run_final_single_point
from p3z_dft.dft_logging import configure_pyscf_logging
from p3z_dft.io_utils import compound_artifact_root, write_artifact_json

PERIODIC_TABLE = Chem.GetPeriodicTable()
DEUTERIUM_ISOTOPE_MASS_U = 2.01410177812
EV_PER_HARTREE = 27.2114

def signed_frequency_cm1(freq_wavenumber):
    freq = np.asarray(freq_wavenumber)
    return np.asarray(np.real(freq) - np.abs(np.imag(freq)), dtype=float)


def ir_lorentz_broadening(v, v_0, width_cm1):
    return 100 * 0.5 / np.log(10) / np.pi * width_cm1 / ((v - v_0) ** 2 + 0.25 * width_cm1 ** 2)


def build_ir_spectrum(freq_cm1, ir_intensities):
    grid = np.arange(IR_SPECTRUM_MIN_CM1, IR_SPECTRUM_MAX_CM1 + 0.5 * IR_SPECTRUM_STEP_CM1, IR_SPECTRUM_STEP_CM1)
    spectrum = np.zeros_like(grid, dtype=float)
    freq = np.asarray(freq_cm1, dtype=float)
    intensities = np.asarray(ir_intensities, dtype=float)
    valid = np.isfinite(freq) & np.isfinite(intensities) & (freq > 0)
    if np.any(valid):
        broadened = ir_lorentz_broadening(grid[:, None], freq[valid][None, :], IR_SPECTRUM_FWHM_CM1)
        spectrum = np.sum(broadened * intensities[valid][None, :], axis=1)
    normalized = spectrum / np.max(spectrum) if np.any(spectrum > 0) else np.zeros_like(spectrum)
    return grid, spectrum, normalized


def default_spin_for_charge(symbols, charge):
    n_electrons = int(sum(PERIODIC_TABLE.GetAtomicNumber(str(sym)) for sym in symbols) - int(charge))
    return 0 if (n_electrons % 2 == 0) else 1


def full_deuteration_masses(symbols, mol_pyscf):
    masses = np.asarray(mol_pyscf.atom_mass_list(isotope_avg=True), dtype=float)
    for atom_idx, symbol in enumerate(symbols):
        if str(symbol) == "H":
            masses[atom_idx] = DEUTERIUM_ISOTOPE_MASS_U
    return masses


def get_h2ao_dipderiv(mf):
    mol = mf.mol
    natm, nao = mol.natm, mol.nao
    h2ao = np.zeros((natm, 3, 3, nao, nao))
    int1e_irp = mol.intor("int1e_irp").reshape(3, 3, nao, nao).swapaxes(0, 1)
    for atom_idx in range(natm):
        _, _, a1, a2 = mol.aoslice_by_atom()[atom_idx]
        h2ao[atom_idx, :, :, :, a1:a2] = int1e_irp[:, :, :, a1:a2]
    h2ao += h2ao.swapaxes(-1, -2)
    return h2ao


def proc_hessian_intermediates(mf_hess, mo_energy=None, mo_coeff=None, mo_occ=None, h1ao_grad=None):
    mf = mf_hess.base
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if h1ao_grad is None:
        h1ao_grad = mf_hess.make_h1(mo_coeff, mo_occ)

    moao1_grad, mo_e1_grad = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao_grad)
    hess_elec = mf_hess.hess_elec(mo1=moao1_grad, mo_e1=mo_e1_grad, h1ao=h1ao_grad)
    hess_nuc = mf_hess.hess_nuc()
    mf_hess.de = hess_elec + hess_nuc

    if isinstance(mo_coeff, (tuple, list)):
        mo1_grad = [None, None]
        for spin_idx in (0, 1):
            mo1_grad[spin_idx] = runtime.lib.einsum(
                "up, uv, Axvi -> Axpi",
                mo_coeff[spin_idx],
                mf.get_ovlp(),
                moao1_grad[spin_idx],
            )
    else:
        mo1_grad = runtime.lib.einsum("up, uv, Axvi -> Axpi", mo_coeff, mf.get_ovlp(), moao1_grad)

    return mf_hess.de, mo1_grad


def compute_harmonic_ir_analysis(mf):
    mf_hess = configure_pyscf_logging(mf.Hessian())
    hessian_tensor, mo1_grad = proc_hessian_intermediates(mf_hess)
    vib_dict = runtime.thermo.harmonic_analysis(mf.mol, hessian_tensor)

    if isinstance(mf.mo_coeff, (tuple, list)):
        h1_dip = [None, None]
        int_r = mf.mol.intor_symmetric("int1e_r")
        for spin_idx in (0, 1):
            h1_dip[spin_idx] = runtime.lib.einsum(
                "tuv, up, vi-> tpi",
                int_r,
                mf.mo_coeff[spin_idx],
                mf.mo_coeff[spin_idx][:, mf.mo_occ[spin_idx] > 0],
            )
    else:
        orbo = mf.mo_coeff[:, mf.mo_occ > 0]
        h1_dip = runtime.lib.einsum("tuv, up, vi-> tpi", mf.mol.intor_symmetric("int1e_r"), mf.mo_coeff, orbo)

    dipole_derivatives = np.zeros((mf.mol.natm, 3, 3))
    for atom_idx in range(mf.mol.natm):
        dipole_derivatives[atom_idx] = np.eye(3) * mf.mol.atom_charge(atom_idx)

    h2ao_dipderiv = get_h2ao_dipderiv(mf)
    rdm1 = mf.make_rdm1()
    if isinstance(mo1_grad, list):
        dipole_derivatives += np.einsum("Axtuv, suv -> Axt", h2ao_dipderiv, rdm1)
        for spin_idx in (0, 1):
            dipole_derivatives -= 2 * np.einsum("tpi, Axpi -> Axt", h1_dip[spin_idx], mo1_grad[spin_idx])
    else:
        dipole_derivatives += np.einsum("Axtuv, uv -> Axt", h2ao_dipderiv, rdm1)
        dipole_derivatives -= 4 * np.einsum("tpi, Axpi -> Axt", h1_dip, mo1_grad)

    normal_modes = np.asarray(vib_dict["norm_mode"], dtype=float)
    q_matrix = normal_modes.reshape(-1, mf.mol.natm * 3)
    dipole_q = np.dot(q_matrix, dipole_derivatives.reshape(-1, 3))

    alpha = physical_constants["fine-structure constant"][0]
    amu = physical_constants["atomic mass constant"][0]
    electron_mass = physical_constants["electron mass"][0]
    avogadro = physical_constants["Avogadro constant"][0]
    bohr_radius = physical_constants["Bohr radius"][0]
    unit_kmmol = alpha ** 2 * (1e-3 / amu) * electron_mass * avogadro * np.pi * bohr_radius / 3
    ir_intensities = unit_kmmol * np.einsum("qt, qt -> q", dipole_q, dipole_q)

    return {
        "hessian_au": np.asarray(hessian_tensor, dtype=float),
        "vib_dict": vib_dict,
        "dipole_derivatives_au": dipole_derivatives,
        "ir_intensities_km_mol": np.asarray(ir_intensities, dtype=float),
    }


def compute_vibrational_outputs(mol_pyscf, compound_id):
    artifact_dir = compound_artifact_root(compound_id) / "vibrations"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mf_vib, _ = run_final_single_point(
        mol_pyscf,
        use_d3=False,
        scf_conv_tol=VIBRATION_SCF_CONV_TOL,
        max_scf_cycles=VIBRATION_SCF_MAX_CYCLES,
        grid_level=VIBRATION_GRID_LEVEL,
    )
    if not getattr(mf_vib, "converged", False):
        raise RuntimeError("vibrational reference SCF did not converge")

    analysis = compute_harmonic_ir_analysis(mf_vib)
    vib_dict = analysis["vib_dict"]
    frequencies_cm1 = signed_frequency_cm1(vib_dict["freq_wavenumber"])
    ir_intensities = np.asarray(analysis["ir_intensities_km_mol"], dtype=float)
    normal_modes = np.asarray(vib_dict["norm_mode"], dtype=float)

    spectrum_path = None
    if COMPUTE_IR_SPECTRA:
        grid_cm1, raw_spectrum, normalized_spectrum = build_ir_spectrum(frequencies_cm1, ir_intensities)
        spectrum_path = artifact_dir / "ir_spectrum.npz"
        np.savez_compressed(
            spectrum_path,
            wavenumber_cm1=grid_cm1,
            absorption_coefficient_l_mol_cm=raw_spectrum,
            normalized_absorption=normalized_spectrum,
            line_fwhm_cm1=np.asarray([IR_SPECTRUM_FWHM_CM1], dtype=float),
        )

    write_artifact_json(artifact_dir / "frequencies_cm1.json", frequencies_cm1.tolist())
    write_artifact_json(artifact_dir / "ir_intensities_km_mol.json", ir_intensities.tolist())
    np.savez_compressed(
        artifact_dir / "normal_modes.npz",
        norm_mode=normal_modes,
        reduced_mass_au=np.asarray(vib_dict["reduced_mass"], dtype=float),
        force_const_au=np.asarray(vib_dict["force_const_au"], dtype=float),
        hessian_au=np.asarray(analysis["hessian_au"], dtype=float),
        dipole_derivatives_au=np.asarray(analysis["dipole_derivatives_au"], dtype=float),
    )

    summary_path = write_artifact_json(
        artifact_dir / "summary.json",
        {
            "compound_id": str(compound_id),
            "method": f"{DFT_FUNCTIONAL.upper()}/{DFT_BASIS}",
            "hessian_includes_d3": False,
            "n_modes": int(len(frequencies_cm1)),
            "n_imaginary_modes": int(np.count_nonzero(frequencies_cm1 < 0)),
            "min_frequency_cm1": float(np.min(frequencies_cm1)) if len(frequencies_cm1) else None,
            "max_ir_intensity_km_mol": float(np.max(ir_intensities)) if len(ir_intensities) else None,
            "ir_intensity_sum_km_mol": float(np.sum(ir_intensities)) if len(ir_intensities) else None,
            "ir_spectrum_path": str(spectrum_path) if spectrum_path is not None else None,
        },
    )

    return (
        {
            "dft_vibrational_status": "success",
            "dft_n_modes": int(len(frequencies_cm1)),
            "dft_n_imaginary_modes": int(np.count_nonzero(frequencies_cm1 < 0)),
            "dft_min_frequency_cm1": float(np.min(frequencies_cm1)) if len(frequencies_cm1) else np.nan,
            "dft_max_ir_intensity": float(np.max(ir_intensities)) if len(ir_intensities) else np.nan,
            "dft_ir_intensity_sum": float(np.sum(ir_intensities)) if len(ir_intensities) else np.nan,
            "vibration_artifact_dir": str(artifact_dir),
            "ir_spectrum_artifact_path": str(spectrum_path) if spectrum_path is not None else None,
        },
        {
            "artifact_dir": artifact_dir,
            "summary_path": summary_path,
            "analysis": analysis,
            "frequencies_cm1": frequencies_cm1,
        },
    )


def compute_full_deuteration_outputs(compound_id, mol_pyscf, vib_stage):
    artifact_dir = compound_artifact_root(compound_id) / "isotopes" / "full_deuteration"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    symbols = [mol_pyscf.atom_symbol(i) for i in range(mol_pyscf.natm)]
    if not any(symbol == "H" for symbol in symbols):
        write_artifact_json(
            artifact_dir / "summary.json",
            {
                "compound_id": str(compound_id),
                "status": "skipped_no_hydrogen",
                "policy": "full_deuteration",
            },
        )
        return {
            "hd_shift_status": "skipped_no_hydrogen",
            "hd_shift_mean_cm1": np.nan,
            "hd_shift_max_cm1": np.nan,
            "hd_shift_min_cm1": np.nan,
            "hd_shift_artifact_dir": str(artifact_dir),
        }

    deuterated_masses = full_deuteration_masses(symbols, mol_pyscf)
    hd_vib_dict = runtime.thermo.harmonic_analysis(mol_pyscf, vib_stage["analysis"]["hessian_au"], mass=deuterated_masses)
    neutral_freq = np.asarray(vib_stage["frequencies_cm1"], dtype=float)
    deuterated_freq = signed_frequency_cm1(hd_vib_dict["freq_wavenumber"])
    shifts_cm1 = neutral_freq - deuterated_freq

    write_artifact_json(artifact_dir / "mode_shifts_cm1.json", shifts_cm1.tolist())
    np.savez_compressed(
        artifact_dir / "full_deuteration_modes.npz",
        neutral_freq_cm1=neutral_freq,
        deuterated_freq_cm1=deuterated_freq,
        shift_cm1=shifts_cm1,
        deuterated_reduced_mass_au=np.asarray(hd_vib_dict["reduced_mass"], dtype=float),
        deuterated_norm_mode=np.asarray(hd_vib_dict["norm_mode"], dtype=float),
    )
    write_artifact_json(
        artifact_dir / "summary.json",
        {
            "compound_id": str(compound_id),
            "status": "success",
            "policy": "full_deuteration",
            "n_deuterated_sites": int(sum(symbol == "H" for symbol in symbols)),
            "shift_definition": "neutral_frequency_cm1 - deuterated_frequency_cm1",
            "shift_mean_cm1": float(np.mean(shifts_cm1)) if len(shifts_cm1) else None,
            "shift_max_cm1": float(np.max(shifts_cm1)) if len(shifts_cm1) else None,
            "shift_min_cm1": float(np.min(shifts_cm1)) if len(shifts_cm1) else None,
        },
    )

    return {
        "hd_shift_status": "success",
        "hd_shift_mean_cm1": float(np.mean(shifts_cm1)) if len(shifts_cm1) else np.nan,
        "hd_shift_max_cm1": float(np.max(shifts_cm1)) if len(shifts_cm1) else np.nan,
        "hd_shift_min_cm1": float(np.min(shifts_cm1)) if len(shifts_cm1) else np.nan,
        "hd_shift_artifact_dir": str(artifact_dir),
    }


def compute_vertical_ea_outputs(symbols, coords, neutral_charge, neutral_spin, compound_id):
    artifact_dir = compound_artifact_root(compound_id) / "anion"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    neutral_ref = build_pyscf_molecule_from_geometry(symbols, coords, charge=neutral_charge, spin=neutral_spin)
    neutral_mf, neutral_energy = run_final_single_point(
        neutral_ref,
        use_d3=USE_D3,
        scf_conv_tol=ANION_SCF_CONV_TOL,
        max_scf_cycles=ANION_SCF_MAX_CYCLES,
        grid_level=ANION_GRID_LEVEL,
    )
    if not getattr(neutral_mf, "converged", False):
        raise RuntimeError("neutral fixed-geometry reference SCF did not converge")

    anion_charge = int(neutral_charge) - 1
    anion_spin = default_spin_for_charge(symbols, anion_charge)
    anion_mol = build_pyscf_molecule_from_geometry(symbols, coords, charge=anion_charge, spin=anion_spin)
    anion_mf, anion_energy = run_final_single_point(
        anion_mol,
        use_d3=USE_D3,
        scf_conv_tol=ANION_SCF_CONV_TOL,
        max_scf_cycles=ANION_SCF_MAX_CYCLES,
        grid_level=ANION_GRID_LEVEL,
    )
    if not getattr(anion_mf, "converged", False):
        raise RuntimeError("anion fixed-geometry SCF did not converge")

    vertical_ea_ev = float((float(neutral_energy) - float(anion_energy)) * EV_PER_HARTREE)
    summary_path = write_artifact_json(
        artifact_dir / "vertical_ea.json",
        {
            "compound_id": str(compound_id),
            "status": "success",
            "neutral_charge": int(neutral_charge),
            "neutral_spin": int(neutral_spin),
            "anion_charge": int(anion_charge),
            "anion_spin": int(anion_spin),
            "neutral_energy_hartree": float(neutral_energy),
            "anion_energy_hartree": float(anion_energy),
            "vertical_ea_ev": vertical_ea_ev,
            "definition": "(E_neutral_fixed - E_anion_fixed) * 27.2114",
        },
    )
    return {
        "dft_vertical_ea_status": "success",
        "dft_vertical_ea_ev": vertical_ea_ev,
        "anion_artifact_path": str(summary_path),
    }

