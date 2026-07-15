from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

PERIODIC_TABLE = Chem.GetPeriodicTable()

def infer_rdkit_charge_spin_multiplicity(mol_3d):
    """Infer formal charge, PySCF spin/xtB UHF, and multiplicity from an RDKit Mol."""
    charge = int(Chem.GetFormalCharge(mol_3d))
    n_electrons = int(sum(atom.GetAtomicNum() for atom in mol_3d.GetAtoms()) - charge)
    radical_electrons = int(sum(atom.GetNumRadicalElectrons() for atom in mol_3d.GetAtoms()))

    spin = radical_electrons
    if spin == 0 and n_electrons % 2:
        spin = 1
    if spin % 2 != n_electrons % 2:
        spin = 1 if n_electrons % 2 else 0

    multiplicity = int(spin + 1)
    symbols = [atom.GetSymbol() for atom in mol_3d.GetAtoms()]
    validate_xtb_charge_multiplicity(symbols, charge=charge, multiplicity=multiplicity)
    return charge, spin, multiplicity


def validate_xtb_charge_multiplicity(symbols, charge, multiplicity):
    try:
        charge = int(charge)
        multiplicity = int(multiplicity)
    except Exception as exc:
        raise ValueError(f"charge and multiplicity must be integers: {exc}") from exc

    if multiplicity < 1:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")

    atomic_numbers = []
    for symbol in symbols:
        atomic_number = int(PERIODIC_TABLE.GetAtomicNumber(str(symbol)))
        if atomic_number <= 0:
            raise ValueError(f"unknown element symbol for xTB: {symbol!r}")
        atomic_numbers.append(atomic_number)

    n_electrons = int(sum(atomic_numbers) - charge)
    uhf = int(multiplicity - 1)
    if n_electrons <= 0:
        raise ValueError(f"invalid electron count {n_electrons} for charge={charge}")
    if uhf < 0:
        raise ValueError(f"invalid unpaired electron count {uhf}")
    if uhf > n_electrons:
        raise ValueError(f"unpaired electron count {uhf} exceeds electron count {n_electrons}")
    if (n_electrons % 2) != (uhf % 2):
        raise ValueError(
            f"electron parity mismatch: electrons={n_electrons}, multiplicity={multiplicity}, uhf={uhf}"
        )
    return uhf, n_electrons


def rdkit_symbols_and_coordinates(mol_3d):
    conf = mol_3d.GetConformer()
    symbols = [mol_3d.GetAtomWithIdx(i).GetSymbol() for i in range(mol_3d.GetNumAtoms())]
    coords = np.asarray(conf.GetPositions(), dtype=float)
    if coords.shape != (len(symbols), 3) or not np.isfinite(coords).all():
        raise ValueError(f"invalid RDKit coordinates: shape={coords.shape}")
    return symbols, coords


def copy_rdkit_mol_with_coordinates(mol_3d, coordinates_angstrom):
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (mol_3d.GetNumAtoms(), 3) or not np.isfinite(coords).all():
        raise ValueError(f"cannot update RDKit coordinates with shape={coords.shape}")

    mol_copy = Chem.Mol(mol_3d)
    conf = mol_copy.GetConformer()
    for atom_idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
    return mol_copy


def rdkit_charge_and_spin(mol_3d):
    """Infer PySCF charge and spin from the same RDKit logic used for xTB."""
    charge, spin, _multiplicity = infer_rdkit_charge_spin_multiplicity(mol_3d)
    return charge, spin

