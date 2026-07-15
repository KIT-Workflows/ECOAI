from __future__ import annotations

import time
import traceback

import numpy as np

from p3z_dft import config
from p3z_dft import dft_logging
from p3z_dft import runtime
from p3z_dft.config import (
    ALLOW_SINGLE_POINT_FALLBACK,
    DFT_BASIS,
    DFT_FUNCTIONAL,
    GEOMOPT_GRAD_TOL,
    GEOMOPT_MAX_STEPS,
    MAX_SCF_CYCLES,
    PYSCF_MAX_MEMORY,
    PYSCF_VERBOSE,
    SCF_CONV_TOL,
)
from p3z_dft.dft_logging import configure_pyscf_logging, dft_debug, dft_status
from p3z_dft.rdkit_geom import rdkit_charge_and_spin, rdkit_symbols_and_coordinates

EV_PER_HARTREE = 27.2114

def build_pyscf_molecule_from_geometry(symbols, coordinates_angstrom, *, charge, spin, basis=None):
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(symbols), 3):
        raise ValueError(f"invalid geometry shape for PySCF build: {coords.shape}")
    atom_lines = [
        f"{sym} {x:.10f} {y:.10f} {z:.10f}"
        for sym, (x, y, z) in zip(symbols, coords)
    ]
    mol_pyscf = runtime.gto.M(
        atom="; ".join(atom_lines),
        basis=basis or DFT_BASIS,
        charge=int(charge),
        spin=int(spin),
        unit="Angstrom",
        verbose=0 if dft_logging._DFT_ACTIVE_LOG_HANDLE is not None else PYSCF_VERBOSE,
        max_memory=PYSCF_MAX_MEMORY,
    )
    return configure_pyscf_logging(mol_pyscf)


def build_pyscf_molecule(mol_3d, *, charge=None, spin=None, basis=None):
    """Build a PySCF Mole from the RDKit conformer coordinates in angstrom."""
    symbols, positions = rdkit_symbols_and_coordinates(mol_3d)
    inferred_charge, inferred_spin = rdkit_charge_and_spin(mol_3d)
    charge = inferred_charge if charge is None else int(charge)
    spin = inferred_spin if spin is None else int(spin)
    dft_debug(f"  [DFT] build PySCF molecule: atoms={mol_3d.GetNumAtoms()} charge={charge} spin={spin}")
    return build_pyscf_molecule_from_geometry(symbols, positions, charge=charge, spin=spin, basis=basis)


def build_mean_field(mol_pyscf, *, use_d3=None, scf_conv_tol=None, max_scf_cycles=None, grid_level=3):
    """Create a DFT mean-field object, wrapped with D3 when requested."""
    use_d3 = (config.USE_D3 and runtime.D3_AVAILABLE) if use_d3 is None else bool(use_d3 and runtime.D3_AVAILABLE)
    if mol_pyscf.spin == 0:
        mf = runtime.pyscf_dft.RKS(mol_pyscf)
    else:
        mf = runtime.pyscf_dft.UKS(mol_pyscf)

    mf.xc = DFT_FUNCTIONAL
    mf.max_cycle = int(MAX_SCF_CYCLES if max_scf_cycles is None else max_scf_cycles)
    mf.conv_tol = float(SCF_CONV_TOL if scf_conv_tol is None else scf_conv_tol)
    mf.max_memory = PYSCF_MAX_MEMORY
    mf.verbose = PYSCF_VERBOSE
    mf.grids.level = int(grid_level)
    mf = configure_pyscf_logging(mf)

    if use_d3:
        mf = runtime.d3pyscf.energy(mf)
        mf = configure_pyscf_logging(mf)

    return mf


def flatten_mo_energies_and_occ(mf):
    """Return sorted orbital energies and occupations for RKS or UKS objects."""
    mo_energies = mf.mo_energy
    mo_occ = mf.mo_occ

    if isinstance(mo_energies, (tuple, list)):
        energies = np.concatenate([np.asarray(part, dtype=float) for part in mo_energies])
        occ = np.concatenate([np.asarray(part, dtype=float) for part in mo_occ])
    else:
        energies = np.asarray(mo_energies, dtype=float)
        occ = np.asarray(mo_occ, dtype=float)

    order = np.argsort(energies)
    return energies[order], occ[order]


def run_final_single_point(mol_pyscf, **mf_kwargs):
    """Run final DFT(+D3) single point at the supplied geometry."""
    mf = build_mean_field(mol_pyscf, **mf_kwargs)
    total_energy = mf.kernel()
    return mf, total_energy


def copy_mol_with_bohr_coords(template_mol, coords_bohr):
    """Return a Mole copy with updated Cartesian coordinates in Bohr."""
    mol_new = template_mol.copy()
    mol_new.set_geom_(np.asarray(coords_bohr, dtype=float).reshape((-1, 3)), unit="Bohr", inplace=True)
    return mol_new


def optimize_geometry_scipy(initial_mol, compound_id=None):
    """
    Optimize geometry with SciPy using PySCF(+D3) analytic nuclear gradients.

    Coordinates are optimized in Bohr, energies are Hartree, and gradients are Hartree/Bohr.
    This avoids external geometry optimizer packages that are brittle in Colab/Python 3.12.
    """
    x0 = initial_mol.atom_coords(unit="Bohr").reshape(-1)
    cache = {}
    eval_counter = {"n": 0}
    label = compound_id or "molecule"
    dft_debug(
        f"  [OPT] [{label}] start: atoms={initial_mol.natm}, "
        f"basis={DFT_BASIS}, D3={config.USE_D3 and runtime.D3_AVAILABLE}, maxiter={GEOMOPT_MAX_STEPS}"
    )

    def evaluate(x):
        key = np.asarray(x, dtype=float).tobytes()
        if key in cache:
            return cache[key]

        eval_counter["n"] += 1
        eval_id = eval_counter["n"]
        eval_start = time.time()
        mol_current = copy_mol_with_bohr_coords(initial_mol, x)
        mf_current = build_mean_field(mol_current)
        energy = float(mf_current.kernel())
        scf_converged = bool(getattr(mf_current, "converged", False))
        if not scf_converged:
            raise RuntimeError("SCF did not converge during geometry optimization")

        grad_method = configure_pyscf_logging(mf_current.nuc_grad_method())
        grad = np.asarray(grad_method.kernel(), dtype=float).reshape(-1)
        max_grad = float(np.max(np.abs(grad)))
        elapsed = time.time() - eval_start
        dft_debug(
            f"  [OPT] [{label}] eval={eval_id:03d} "
            f"E={energy:.10f} Ha max|grad|={max_grad:.3e} Ha/Bohr "
            f"SCF={scf_converged} time={elapsed:.1f}s"
        )
        cache.clear()
        cache[key] = (energy, grad)
        return cache[key]

    def objective(x):
        return evaluate(x)[0]

    def gradient(x):
        return evaluate(x)[1]

    result = runtime.minimize(
        objective,
        x0,
        jac=gradient,
        method="L-BFGS-B",
        options={
            "maxiter": int(GEOMOPT_MAX_STEPS),
            "gtol": float(GEOMOPT_GRAD_TOL),
            "ftol": 1e-9,
            "maxls": 20,
        },
    )

    mol_optimized = copy_mol_with_bohr_coords(initial_mol, result.x)
    max_grad = float(np.max(np.abs(result.jac))) if result.jac is not None else np.nan
    opt_info = {
        "success": bool(result.success) or max_grad <= GEOMOPT_GRAD_TOL,
        "message": str(result.message),
        "n_iterations": int(getattr(result, "nit", 0)),
        "energy_hartree": float(result.fun),
        "max_gradient_hartree_per_bohr": max_grad,
        "n_evaluations": int(eval_counter["n"]),
    }
    dft_debug(
        f"  [OPT] [{label}] done: success={opt_info['success']} "
        f"nit={opt_info['n_iterations']} neval={opt_info['n_evaluations']} "
        f"max|grad|={opt_info['max_gradient_hartree_per_bohr']:.3e} message={opt_info['message']}"
    )
    return mol_optimized, opt_info

