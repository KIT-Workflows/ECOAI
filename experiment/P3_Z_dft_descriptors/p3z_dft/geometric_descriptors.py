from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

ATOMIC_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'P': 30.974, 'S': 32.06, 'Cl': 35.45,
    'Br': 79.904, 'I': 126.90, 'Si': 28.085, 'B': 10.81,
    'K': 39.098, 'Ca': 40.078, 'Na': 22.990, 'Mg': 24.305,
}

ARBALIGN_AVAILABLE = False
try:
    from arbalign import ArbalignContext
    ARBALIGN_AVAILABLE = True
except ImportError:
    pass

def get_atomic_mass(symbol: str) -> float:
    """Get atomic mass for element symbol. Defaults to heavy atom approximation."""
    return ATOMIC_MASSES.get(symbol, 12.0)


def extract_optimized_geometry(mol_3d, mol_pyscf):
    """
    Extract final PySCF geometry in angstroms.

    PySCF stores coordinates internally in Bohr, so unit="Angstrom" is required here.
    The RDKit argument is retained for backward compatibility with earlier cells.
    """
    symbols = np.array([mol_pyscf.atom_symbol(i) for i in range(mol_pyscf.natm)])
    coords = mol_pyscf.atom_coords(unit="Angstrom")
    return symbols, np.array(coords, dtype=np.float64)


def compute_hoppe_descriptors(
    symbols: np.ndarray,
    coordinates: np.ndarray,
    tolerance: float = 1.0e-4,
    max_iterations: int = 1000
) -> Dict:
    """
    Compute Hoppe average bond length and effective coordination number.

    Parameters:
        symbols: Array of element symbols (N,)
        coordinates: Cartesian coordinates in angstroms (N, 3)
        tolerance: Convergence threshold in angstroms
        max_iterations: Maximum iterations for self-consistency

    Returns:
        dict with keys:
            - d_av_per_atom_A: Average bond length per atom (N,)
            - ECN_per_atom: Effective coordination number per atom (N,)
            - d_av_A: Structure average (float)
            - ECN: Structure average (float)
            - hoppe_converged: Convergence flag (bool)
            - hoppe_iterations_max: Max iterations reached (int)
            - hoppe_status: Status message (str)
    """
    n_atoms = len(symbols)
    coordinates = np.array(coordinates, dtype=np.float64)

    result = {
        "d_av_per_atom_A": np.full(n_atoms, np.nan),
        "ECN_per_atom": np.full(n_atoms, np.nan),
        "d_av_A": np.nan,
        "ECN": np.nan,
        "hoppe_converged": False,
        "hoppe_iterations_max": 0,
        "hoppe_status": "pending"
    }

    # Edge case: single atom
    if n_atoms == 1:
        result["hoppe_status"] = "single_atom"
        result["hoppe_converged"] = True
        return result

    # Compute all pairwise distances
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]  # (N, N, 3)
    distances = np.linalg.norm(diff, axis=2)  # (N, N)

    d_av_i = np.zeros(n_atoms)
    converged_flags = np.zeros(n_atoms, dtype=bool)

    for i in range(n_atoms):
        # Get distances to all other atoms (exclude self)
        d_other = distances[i, np.arange(n_atoms) != i]

        if len(d_other) == 0:
            converged_flags[i] = True
            continue

        # Initial guess: nearest-neighbor distance
        d_av_i[i] = np.min(d_other)

        # Self-consistent iteration
        for iteration in range(max_iterations):
            d_old = d_av_i[i]

            # Compute weights
            ratio = d_other / d_av_i[i]
            weights = np.exp(1.0 - ratio**6)

            # Update
            d_av_i[i] = np.sum(d_other * weights) / np.sum(weights)

            # Check convergence
            if np.abs(d_av_i[i] - d_old) < tolerance:
                converged_flags[i] = True
                result["hoppe_iterations_max"] = max(result["hoppe_iterations_max"], iteration + 1)
                break
        else:
            # Did not converge within max_iterations
            import warnings
            warnings.warn(
                f"Atom {i} ({symbols[i]}) did not converge in Hoppe iteration "
                f"(max_iterations={max_iterations})"
            )

    # Compute final ECN values
    ecn_i = np.zeros(n_atoms)
    for i in range(n_atoms):
        d_other = distances[i, np.arange(n_atoms) != i]
        ratio = d_other / d_av_i[i]
        weights = np.exp(1.0 - ratio**6)
        ecn_i[i] = np.sum(weights)

    result["d_av_per_atom_A"] = d_av_i
    result["ECN_per_atom"] = ecn_i
    result["d_av_A"] = float(np.mean(d_av_i))
    result["ECN"] = float(np.mean(ecn_i))
    result["hoppe_converged"] = bool(np.all(converged_flags))

    if result["hoppe_converged"]:
        result["hoppe_status"] = "converged"
    else:
        unconverged_count = (~converged_flags).sum()
        result["hoppe_status"] = f"unconverged ({unconverged_count}/{n_atoms} atoms)"

    return result


def compute_radius_of_gyration(symbols: np.ndarray, coordinates: np.ndarray) -> float:
    """
    Compute radius of gyration using mass-weighted center of mass.

    Parameters:
        symbols: Array of element symbols (N,)
        coordinates: Cartesian coordinates in angstroms (N, 3)

    Returns:
        Rg in angstroms
    """
    coordinates = np.array(coordinates, dtype=np.float64)
    masses = np.array([get_atomic_mass(s) for s in symbols], dtype=np.float64)

    # Mass-weighted center of mass
    com = np.sum(masses[:, np.newaxis] * coordinates, axis=0) / np.sum(masses)

    # Unweighted average of squared displacements
    displacements = coordinates - com
    distances_sq = np.sum(displacements**2, axis=1)
    rg = np.sqrt(np.mean(distances_sq))

    return float(rg)


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute optimal rotation matrix (Kabsch algorithm) with determinant +1.

    Parameters:
        P: Reference coordinates (N, 3)
        Q: Coordinates to rotate (N, 3)

    Returns:
        (R, rmsd) where R is rotation matrix with det(R)=+1, rmsd is RMSD in angstroms
    """
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)

    # Center both
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)

    # SVD
    H = Q_centered.T @ P_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # RMSD
    Q_rotated = Q_centered @ R.T
    rmsd = np.sqrt(np.mean(np.sum((P_centered - Q_rotated)**2, axis=1)))

    return R, rmsd


def compute_element_constrained_assignment(
    P: np.ndarray,
    Q: np.ndarray,
    symbols_P: np.ndarray,
    symbols_Q: np.ndarray
) -> np.ndarray:
    """
    Find optimal atom assignment respecting element types (Hungarian algorithm).

    Returns:
        assignment array where assignment[i] = j means atom i in Q maps to atom j in P
    """
    n = len(symbols_P)
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Allow assignment only if same element
            if symbols_P[j] == symbols_Q[i]:
                dist = np.linalg.norm(P[j] - Q[i])
                cost_matrix[i, j] = dist
            else:
                cost_matrix[i, j] = 1e10  # Prohibitive cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = np.full(n, -1)
    assignment[row_ind] = col_ind

    return assignment


def compute_hcm(
    symbols: np.ndarray,
    coordinates: np.ndarray,
    tolerance: float = 1.0e-6
) -> Dict:
    """
    Compute Hausdorff chirality measure.

    Parameters:
        symbols: Array of element symbols (N,)
        coordinates: Cartesian coordinates in angstroms (N, 3)
        tolerance: Tolerance for dQ ~ 0 check

    Returns:
        dict with keys:
            - HCM: Hausdorff chirality measure (dimensionless)
            - HD_A: Hausdorff distance (angstroms)
            - dQ_A: Maximum internal distance (angstroms)
            - alignment_RMSD_A: RMSD after alignment (angstroms)
            - hcm_status: Status message
    """
    coordinates = np.array(coordinates, dtype=np.float64)
    n_atoms = len(symbols)

    result = {
        "HCM": np.nan,
        "HD_A": np.nan,
        "dQ_A": np.nan,
        "alignment_RMSD_A": np.nan,
        "hcm_status": "pending"
    }

    # Edge cases
    if n_atoms < 2:
        result["hcm_status"] = "too_few_atoms"
        return result

    # Center at centroid (geometric center, not mass-weighted)
    coords_centered = coordinates - np.mean(coordinates, axis=0)

    # Generate mirror image (negate x coordinate)
    mirror = coords_centered.copy()
    mirror[:, 0] *= -1

    # Maximum internal distance
    distances = squareform(pdist(coords_centered))
    dQ = np.max(distances)
    result["dQ_A"] = float(dQ)

    if dQ < tolerance:
        result["hcm_status"] = "zero_size"
        return result

    # Align mirror to original
    try:
        if ARBALIGN_AVAILABLE:
            # Use ArbAlign if available
            context = ArbalignContext(coords_centered, mirror, symbols, symbols)
            rmsd = context.compute_alignment()
            mirror_aligned = context.get_other_coords()  # Aligned coordinates
            result["alignment_RMSD_A"] = float(rmsd)
        else:
            # Fallback: element-constrained Hungarian + Kabsch
            assignment = compute_element_constrained_assignment(
                coords_centered, mirror, symbols, symbols
            )

            if np.any(assignment < 0):
                result["hcm_status"] = "invalid_assignment"
                return result

            # Permute mirror to match assignment
            mirror_assigned = mirror[np.argsort(assignment)]
            R, rmsd = kabsch_rotation(coords_centered, mirror_assigned)
            mirror_aligned = (mirror_assigned - np.mean(mirror_assigned, axis=0)) @ R.T
            mirror_aligned += np.mean(coords_centered, axis=0)
            result["alignment_RMSD_A"] = float(rmsd)

        # Compute symmetric Hausdorff distance (element-aware)
        h_forward = 0.0
        for i in range(n_atoms):
            # Find minimum distance to any atom of same element in mirror
            same_element_mask = symbols == symbols[i]
            if np.any(same_element_mask):
                distances_to_mirror = np.linalg.norm(mirror_aligned[same_element_mask] - coords_centered[i], axis=1)
                h_forward = max(h_forward, np.min(distances_to_mirror))

        h_backward = 0.0
        for i in range(n_atoms):
            same_element_mask = symbols == symbols[i]
            if np.any(same_element_mask):
                distances_to_original = np.linalg.norm(coords_centered[same_element_mask] - mirror_aligned[i], axis=1)
                h_backward = max(h_backward, np.min(distances_to_original))

        hd = max(h_forward, h_backward)
        result["HD_A"] = float(hd)
        result["HCM"] = float(hd / dQ)
        result["hcm_status"] = "success"

    except Exception as e:
        result["hcm_status"] = f"failed: {str(e)[:100]}"

    return result

