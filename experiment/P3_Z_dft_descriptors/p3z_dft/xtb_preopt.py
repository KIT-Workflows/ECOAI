from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from p3z_dft.config import XTB_CALCULATIONS_DIR
from p3z_dft import runtime
from p3z_dft.dft_logging import dft_debug
from p3z_dft.io_utils import read_xyz, safe_xtb_name, write_xyz
from p3z_dft.rdkit_geom import validate_xtb_charge_multiplicity

PERIODIC_TABLE = Chem.GetPeriodicTable()
XTB_METHOD = "GFN2-xTB"
XTB_CONVERGENCE_PHRASE = "GEOMETRY OPTIMIZATION CONVERGED"

def xtb_note(message):
    """Log an xTB message through the DFT logger when available."""
    logger = globals().get("dft_debug")
    if callable(logger):
        logger(message)
    else:
        print(message, flush=True)


def validate_xtb_output_geometry(input_symbols, output_symbols, output_coords):
    if list(input_symbols) != list(output_symbols):
        raise ValueError("xTB optimized geometry changed atom order or elements")
    output_coords = np.asarray(output_coords, dtype=float)
    if output_coords.shape != (len(input_symbols), 3):
        raise ValueError(f"xTB optimized coordinate shape mismatch: {output_coords.shape}")
    if not np.isfinite(output_coords).all():
        raise ValueError("xTB optimized coordinates contain non-finite values")
    return output_coords


def xtb_input_geometry_hash(symbols, coordinates_angstrom):
    coords = np.asarray(coordinates_angstrom, dtype=float)
    payload = {
        "symbols": list(map(str, symbols)),
        "coordinates_angstrom": np.round(coords, 10).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def kabsch_rmsd_same_order(coords_a, coords_b):
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"RMSD coordinate shape mismatch: {a.shape} vs {b.shape}")
    if len(a) == 0:
        return np.nan
    a_centered = a - a.mean(axis=0)
    b_centered = b - b.mean(axis=0)
    covariance = b_centered.T @ a_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    b_aligned = b_centered @ rotation
    diff = a_centered - b_aligned
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def parse_xtb_scalar(patterns, text):
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            value = matches[-1]
            if isinstance(value, tuple):
                value = value[-1]
            try:
                return float(value.replace("D", "E"))
            except Exception:
                continue
    return np.nan


def parse_xtb_cycles(text):
    patterns = [
        r"GEOMETRY OPTIMIZATION CONVERGED\s+AFTER\s+(\d+)\s+CYCLES",
        r"OPTIMIZATION\s+CONVERGED\s+AFTER\s+(\d+)\s+CYCLES",
        r"cycle\s+(\d+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return int(matches[-1])
    return None


def parse_xtb_outputs(stdout_text, opt_log_text=""):
    text = "\n".join([stdout_text or "", opt_log_text or ""])
    energy = parse_xtb_scalar([
        r"TOTAL ENERGY\s+(-?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
        r"total E\s*=\s*(-?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
    ], text)
    gradient_norm = parse_xtb_scalar([
        r"gradient norm\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
        r"gnorm\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
        r"\|grad\|\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
    ], text)
    return {
        "energy_Eh": float(energy) if np.isfinite(energy) else np.nan,
        "gradient_norm_Eh_per_bohr": float(gradient_norm) if np.isfinite(gradient_norm) else np.nan,
        "cycles": parse_xtb_cycles(text),
    }


def xtb_result_fields(xtb_result):
    return {
        "xtb_method": xtb_result.get("method", XTB_METHOD),
        "xtb_version": xtb_result.get("xtb_version"),
        "xtb_charge": xtb_result.get("charge"),
        "xtb_multiplicity": xtb_result.get("multiplicity"),
        "xtb_uhf": xtb_result.get("uhf"),
        "xtb_solvent": xtb_result.get("solvent"),
        "xtb_opt_level": xtb_result.get("optimization_level"),
        "xtb_converged": bool(xtb_result.get("converged", False)),
        "xtb_energy_Eh": xtb_result.get("energy_Eh"),
        "xtb_gradient_norm_Eh_per_bohr": xtb_result.get("gradient_norm_Eh_per_bohr"),
        "xtb_cycles": xtb_result.get("optimization_cycles"),
        "xtb_wall_time_s": xtb_result.get("wall_time_seconds"),
        "mmff_to_xtb_rmsd_A": xtb_result.get("mmff_to_xtb_rmsd_A"),
        "xtb_input_geometry_path": xtb_result.get("input_geometry_path"),
        "xtb_optimized_geometry_path": xtb_result.get("output_geometry_path"),
        "xtb_status": xtb_result.get("status"),
        "xtb_message": xtb_result.get("message"),
    }


def write_xtb_status(status_path, status):
    serializable = {k: v for k, v in status.items() if k != "optimized_coordinates_angstrom"}
    with Path(status_path).open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, default=str)


def xtb_failure_result(base_status, category, message, return_code=None):
    result = dict(base_status)
    result.update({
        "converged": False,
        "status": category,
        "message": str(message)[:500],
        "return_code": return_code,
        "output_geometry_path": None,
        "optimized_coordinates_angstrom": None,
    })
    write_xtb_status(result["status_json_path"], result)
    return result


def run_xtb_preoptimization(
    molecule_id,
    symbols,
    coordinates_angstrom,
    charge,
    multiplicity,
    output_root,
    solvent=None,
    optimization_level="tight",
    max_cycles=500,
    timeout_seconds=None,
    overwrite=False,
    executable=None,
    xtb_version=None,
    dry_run=False,
):
    """Run restartable GFN2-xTB geometry pre-optimization in an isolated directory."""
    symbols = list(map(str, symbols))
    coords = np.asarray(coordinates_angstrom, dtype=float)
    safe_id = safe_xtb_name(molecule_id)
    workdir = Path(output_root) / safe_id / "xtb_preopt"
    workdir.mkdir(parents=True, exist_ok=True)

    input_xyz = workdir / "xtb_input.xyz"
    stdout_path = workdir / "xtb_output.log"
    stderr_path = workdir / "xtb_error.log"
    status_path = workdir / "xtb_status.json"
    opt_xyz = workdir / "xtbopt.xyz"
    opt_log = workdir / "xtbopt.log"

    input_hash = xtb_input_geometry_hash(symbols, coords)
    executable = executable or runtime.XTB_EXECUTABLE
    xtb_version = xtb_version or runtime.XTB_VERSION
    solvent = solvent or None

    base_status = {
        "molecule_id": str(molecule_id),
        "xtb_version": xtb_version,
        "method": XTB_METHOD,
        "optimization_level": optimization_level,
        "charge": None,
        "multiplicity": None,
        "uhf": None,
        "solvent": solvent,
        "command": None,
        "converged": False,
        "return_code": None,
        "energy_Eh": np.nan,
        "gradient_norm_Eh_per_bohr": np.nan,
        "optimization_cycles": None,
        "wall_time_seconds": np.nan,
        "input_geometry_hash": input_hash,
        "input_geometry_path": str(input_xyz),
        "output_geometry_path": None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "status_json_path": str(status_path),
        "message": "pending",
        "status": "pending",
        "from_cache": False,
        "mmff_to_xtb_rmsd_A": np.nan,
    }

    try:
        uhf, _n_electrons = validate_xtb_charge_multiplicity(symbols, charge, multiplicity)
        charge = int(charge)
        multiplicity = int(multiplicity)
        max_cycles = int(max_cycles)
        base_status.update({"charge": charge, "multiplicity": multiplicity, "uhf": uhf})
    except Exception as exc:
        return xtb_failure_result(base_status, "invalid_charge_or_multiplicity", exc)

    command = [
        str(executable) if executable else "xtb",
        input_xyz.name,
        "--gfn", "2",
        "--opt", str(optimization_level),
        "--cycles", str(max_cycles),
        "--chrg", str(charge),
        "--uhf", str(uhf),
    ]
    if solvent:
        command.extend(["--alpb", str(solvent)])
    base_status["command"] = command

    def cache_matches(status):
        return (
            bool(status.get("converged"))
            and status.get("input_geometry_hash") == input_hash
            and status.get("method") == XTB_METHOD
            and status.get("optimization_level") == optimization_level
            and int(status.get("charge")) == charge
            and int(status.get("multiplicity")) == multiplicity
            and int(status.get("uhf")) == uhf
            and (status.get("solvent") or None) == solvent
            and status.get("xtb_version") == xtb_version
        )

    if not overwrite and status_path.exists() and opt_xyz.exists():
        try:
            cached_status = json.loads(status_path.read_text(encoding="utf-8"))
            if cache_matches(cached_status):
                cached_symbols, cached_coords = read_xyz(opt_xyz)
                cached_coords = validate_xtb_output_geometry(symbols, cached_symbols, cached_coords)
                cached_result = dict(base_status)
                cached_result.update(cached_status)
                cached_result.update({
                    "from_cache": True,
                    "optimized_coordinates_angstrom": cached_coords,
                    "output_geometry_path": str(opt_xyz),
                    "message": cached_status.get("message", "reused cached converged xTB geometry"),
                    "status": cached_status.get("status", "success"),
                })
                return cached_result
        except Exception as exc:
            xtb_note(f"  [WARN] [{molecule_id}] Ignoring invalid xTB cache: {type(exc).__name__}: {exc}")

    try:
        write_xyz(input_xyz, symbols, coords, comment=f"{molecule_id} MMFF geometry for GFN2-xTB preoptimization")
    except Exception as exc:
        return xtb_failure_result(base_status, "invalid_input_geometry", exc)

    if executable is None or not Path(str(executable)).exists():
        return xtb_failure_result(base_status, "executable_not_found", "xTB executable not found on PATH")

    if dry_run:
        result = dict(base_status)
        result.update({
            "status": "dry_run",
            "message": "dry run: input XYZ and command prepared; xTB was not executed",
            "return_code": None,
            "wall_time_seconds": 0.0,
        })
        write_xtb_status(status_path, result)
        return result

    start_time = time.time()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            proc = subprocess.run(
                command,
                cwd=workdir,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                check=False,
            )
        wall_time = time.time() - start_time
    except subprocess.TimeoutExpired as exc:
        result = xtb_failure_result(base_status, "timeout", f"xTB timed out after {timeout_seconds} seconds")
        result["wall_time_seconds"] = time.time() - start_time
        write_xtb_status(status_path, result)
        return result
    except Exception as exc:
        result = xtb_failure_result(base_status, "execution_error", f"{type(exc).__name__}: {exc}")
        result["wall_time_seconds"] = time.time() - start_time
        write_xtb_status(status_path, result)
        return result

    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    opt_log_text = opt_log.read_text(encoding="utf-8", errors="replace") if opt_log.exists() else ""
    combined_text = "\n".join([stdout_text, stderr_text, opt_log_text])
    parsed = parse_xtb_outputs(stdout_text, opt_log_text)

    result = dict(base_status)
    result.update({
        "return_code": int(proc.returncode),
        "wall_time_seconds": float(wall_time),
        "energy_Eh": parsed["energy_Eh"],
        "gradient_norm_Eh_per_bohr": parsed["gradient_norm_Eh_per_bohr"],
        "optimization_cycles": parsed["cycles"],
    })

    if proc.returncode != 0:
        category = "xtb_scc_failure" if ("scc" in combined_text.lower() and ("fail" in combined_text.lower() or "conver" in combined_text.lower())) else "xtb_return_code_nonzero"
        result.update({"status": category, "message": f"xTB returned non-zero exit code {proc.returncode}"})
        write_xtb_status(status_path, result)
        return result

    if XTB_CONVERGENCE_PHRASE not in combined_text.upper():
        result.update({"status": "geometry_optimization_not_converged", "message": "xTB output lacks explicit convergence message"})
        write_xtb_status(status_path, result)
        return result

    if not opt_xyz.exists():
        result.update({"status": "missing_xtbopt_xyz", "message": "xTB converged message found but xtbopt.xyz is missing"})
        write_xtb_status(status_path, result)
        return result

    try:
        output_symbols, output_coords = read_xyz(opt_xyz)
        output_coords = validate_xtb_output_geometry(symbols, output_symbols, output_coords)
        rmsd = kabsch_rmsd_same_order(coords, output_coords)
    except Exception as exc:
        result.update({"status": "invalid_optimized_geometry", "message": f"{type(exc).__name__}: {exc}"})
        write_xtb_status(status_path, result)
        return result

    result.update({
        "converged": True,
        "status": "success",
        "message": "GEOMETRY OPTIMIZATION CONVERGED",
        "output_geometry_path": str(opt_xyz),
        "optimized_coordinates_angstrom": output_coords,
        "mmff_to_xtb_rmsd_A": rmsd,
    })
    write_xtb_status(status_path, result)
    return result

