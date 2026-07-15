from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p3z_dft.config import XTB_CALCULATIONS_DIR

def safe_xtb_name(value):
    text = str(value or "molecule")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return safe[:120] or "molecule"


def write_xyz(path, symbols, coordinates_angstrom, comment=""):
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(symbols), 3) or not np.isfinite(coords).all():
        raise ValueError(f"invalid XYZ coordinates: shape={coords.shape}")
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, (x, y, z) in zip(symbols, coords):
            handle.write(f"{symbol:2s} {x: .12f} {y: .12f} {z: .12f}\n")


def read_xyz(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file too short: {path}")
    try:
        n_atoms = int(lines[0].strip())
    except Exception as exc:
        raise ValueError(f"invalid XYZ atom count in {path}: {lines[0]!r}") from exc
    atom_lines = lines[2:2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"XYZ atom count mismatch in {path}: expected {n_atoms}, got {len(atom_lines)}")

    symbols = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"invalid XYZ atom line in {path}: {line!r}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (n_atoms, 3) or not np.isfinite(coords).all():
        raise ValueError(f"invalid XYZ coordinates in {path}")
    return symbols, coords


def compound_artifact_root(compound_id):
    root = XTB_CALCULATIONS_DIR / safe_xtb_name(compound_id or "molecule")
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_artifact_json(path, payload):
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return artifact_path

