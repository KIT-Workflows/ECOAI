#!/usr/bin/env python
"""P3-Z DFT descriptors pipeline (from Celso_P3_Z_dft_descriptors.ipynb)."""

# --- skipped cell 1: env diagnostic ---
# --- skipped cell 2: Colab bootstrap ---
# %% cell 3
import hashlib
import json
import os
import re
import random
import shutil
import subprocess
import tarfile
import time
import traceback
import urllib.request
import warnings
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    from rdkit.Geometry import Point3D
except ModuleNotFoundError as exc:
    if exc.name == "rdkit":
        import sys
        print("[FATAL] RDKit is not installed in the active Jupyter kernel.")
        print(f"        Python executable: {sys.executable}")
        print(f"        Python prefix:     {sys.prefix}")
        print(f"        CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV')}")
        print("        Install RDKit in this exact environment or re-register/select the playground_simstack kernel.")
    raise

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# %% cell 4
import os

# Export the conda default path (profile script for conda shell integration)
os.environ["CONDA_DEFAULT_PATH"] = str(os.path.expanduser("~/miniforge3/etc/profile.d/conda.sh"))
print('Exported CONDA_DEFAULT_PATH:', os.environ["CONDA_DEFAULT_PATH"])

# Set the environment variables inside the notebook (effective for this process)
os.environ["P3Z_INPUT_PARQUET"] = "Datasets/data/curated_molecules_with_splits.parquet"
os.environ["P3Z_CLASSICAL_FEATURES"] = "Datasets/data/features_combined.parquet"

print('Set P3Z_INPUT_PARQUET to:', os.environ["P3Z_INPUT_PARQUET"])
print('Set P3Z_CLASSICAL_FEATURES to:', os.environ["P3Z_CLASSICAL_FEATURES"])

# %% cell 5
# ================================================================
# S1 — Configuration & Environment Detection
# ================================================================

# --- Detect Google Colab ---
IS_COLAB = False
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    pass


def _first_existing_path(candidates):
    """Return the first existing path, otherwise the first candidate."""
    candidates = [Path(p).expanduser() for p in candidates]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _env_path(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _resolve_data_path(env_name, candidates):
    """Resolve an input path from an env var or a list of likely locations."""
    candidates = [Path(p).expanduser() for p in candidates]
    env_value = os.environ.get(env_name)
    if env_value:
        path = Path(env_value).expanduser().resolve()
        return path, [path]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(), [p.resolve() for p in candidates]

    return candidates[0].resolve(), [p.resolve() for p in candidates]


if IS_COLAB:
    # Local Colab storage by default. Set P3Z_USE_DRIVE=1 later if persistent Drive output is needed.
    use_drive = os.environ.get("P3Z_USE_DRIVE", "0") == "1"
    drive_root = Path("/content/drive/MyDrive")
    drive_available = drive_root.exists()

    if use_drive and not drive_available:
        try:
            from google.colab import drive
            drive.mount("/content/drive", timeout_ms=120000)
            drive_available = drive_root.exists()
        except Exception as exc:
            drive_available = drive_root.exists()
            print(f"[WARN] Google Drive mount failed: {type(exc).__name__}: {exc}")
            if not drive_available:
                print("[WARN] Falling back to local Colab storage under /content/P3_Z_dft_descriptors.")

    COLAB_ROOT = (drive_root / "P3_Z_dft_descriptors") if (use_drive and drive_available) else Path("/content/P3_Z_dft_descriptors")
    SCRIPT_DIR = _env_path("P3Z_SCRIPT_DIR", COLAB_ROOT)
    ARTIFACTS_DIR = _env_path("P3Z_ARTIFACTS_DIR", SCRIPT_DIR / "artifacts")
    PLOTS_DIR = _env_path("P3Z_PLOTS_DIR", SCRIPT_DIR / "plots")

    INPUT_PARQUET_CANDIDATES = [
        SCRIPT_DIR / "inputs" / "curated_molecules_with_splits.parquet",
        Path("/content/P3_Z_dft_descriptors/inputs/curated_molecules_with_splits.parquet"),
        Path("/content/inputs/curated_molecules_with_splits.parquet"),
        Path("/content/inpunts/curated_molecules_with_splits.parquet"),
        Path("/content/curated_molecules_with_splits.parquet"),
        Path.cwd() / "curated_molecules_with_splits.parquet",
    ]
    if drive_available:
        INPUT_PARQUET_CANDIDATES.extend([
            drive_root / "P3_Z_dft_descriptors" / "inputs" / "curated_molecules_with_splits.parquet",
            drive_root / "curated_molecules_with_splits.parquet",
        ])

    CLASSICAL_FEATURES_CANDIDATES = [
        SCRIPT_DIR / "inputs" / "features_combined.parquet",
        Path("/content/P3_Z_dft_descriptors/inputs/features_combined.parquet"),
        Path("/content/inputs/features_combined.parquet"),
        Path("/content/inpunts/features_combined.parquet"),
        Path("/content/features_combined.parquet"),
        Path.cwd() / "features_combined.parquet",
    ]
    if drive_available:
        CLASSICAL_FEATURES_CANDIDATES.extend([
            drive_root / "P3_Z_dft_descriptors" / "inputs" / "features_combined.parquet",
            drive_root / "features_combined.parquet",
        ])

    INPUT_PARQUET, INPUT_PARQUET_CANDIDATES = _resolve_data_path("P3Z_INPUT_PARQUET", INPUT_PARQUET_CANDIDATES)
    CLASSICAL_FEATURES, CLASSICAL_FEATURES_CANDIDATES = _resolve_data_path("P3Z_CLASSICAL_FEATURES", CLASSICAL_FEATURES_CANDIDATES)

    print(f"\n[COLAB] Running on Google Colab.")
    print(f"[COLAB] Using Google Drive: {use_drive and drive_available}")
    print(f"[COLAB] Notebook root: {SCRIPT_DIR}")
    if not (use_drive and drive_available):
        print("[COLAB] Local mode: upload inputs to /content/inpunts, /content/inputs, or /content/P3_Z_dft_descriptors/inputs.")
        print("[COLAB] Set P3Z_USE_DRIVE=1 later to use /content/drive/MyDrive/P3_Z_dft_descriptors.")
else:
    # Local: support both full project layout and standalone notebook folder.
    try:
        start_dir = Path(__file__).parent.resolve()
    except NameError:
        start_dir = Path.cwd().resolve()

    project_root = None
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "experiment").is_dir():
            project_root = candidate
            break
        if candidate.name == "experiment":
            project_root = candidate.parent
            break

    if project_root is not None:
        PROJECT_ROOT = project_root
        default_script_dir = PROJECT_ROOT / "experiment" / "P3_Z_dft_descriptors"
        default_input = PROJECT_ROOT / "experiment" / "P2_scaffold_splitting" / "artifacts" / "curated_molecules_with_splits.parquet"
        default_classical = PROJECT_ROOT / "experiment" / "P3_feature_engineering" / "artifacts" / "features_combined.parquet"
    else:
        PROJECT_ROOT = start_dir
        default_script_dir = start_dir
        default_input_candidates = [
            start_dir / "inputs" / "curated_molecules_with_splits.parquet",
            start_dir / "curated_molecules_with_splits.parquet",
            start_dir / "P2_scaffold_splitting" / "artifacts" / "curated_molecules_with_splits.parquet",
        ]
        default_classical_candidates = [
            start_dir / "inputs" / "features_combined.parquet",
            start_dir / "features_combined.parquet",
            start_dir / "P3_feature_engineering" / "artifacts" / "features_combined.parquet",
        ]
        default_input = _first_existing_path(default_input_candidates)
        default_classical = _first_existing_path(default_classical_candidates)

    SCRIPT_DIR = _env_path("P3Z_SCRIPT_DIR", default_script_dir)
    ARTIFACTS_DIR = _env_path("P3Z_ARTIFACTS_DIR", SCRIPT_DIR / "artifacts")
    PLOTS_DIR = _env_path("P3Z_PLOTS_DIR", SCRIPT_DIR / "plots")
    INPUT_PARQUET_CANDIDATES = [default_input, SCRIPT_DIR / "inputs" / "curated_molecules_with_splits.parquet"]
    CLASSICAL_FEATURES_CANDIDATES = [default_classical, SCRIPT_DIR / "inputs" / "features_combined.parquet"]
    INPUT_PARQUET, INPUT_PARQUET_CANDIDATES = _resolve_data_path("P3Z_INPUT_PARQUET", INPUT_PARQUET_CANDIDATES)
    CLASSICAL_FEATURES, CLASSICAL_FEATURES_CANDIDATES = _resolve_data_path("P3Z_CLASSICAL_FEATURES", CLASSICAL_FEATURES_CANDIDATES)

    print(f"\n[LOCAL] Running locally.")
    print(f"[LOCAL] Project root: {PROJECT_ROOT}")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DFT_LOG_DIR = _env_path("P3Z_DFT_LOG_DIR", ARTIFACTS_DIR / "dft_logs")
DFT_LOG_DIR.mkdir(parents=True, exist_ok=True)
XTB_CALCULATIONS_DIR = _env_path("P3Z_XTB_CALCULATIONS_DIR", ARTIFACTS_DIR / "calculations")
XTB_CALCULATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Keep Matplotlib/font caches inside the writable project tree.
MPLCONFIG_DIR = ARTIFACTS_DIR / "_matplotlib"
XDG_CACHE_DIR = ARTIFACTS_DIR / "_cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

print(f"[PATH] Script/output root: {SCRIPT_DIR}")
print(f"[PATH] Input parquet:      {INPUT_PARQUET}")
if not INPUT_PARQUET.exists():
    print("[PATH] Input parquet not found yet. Checked:")
    for candidate in INPUT_PARQUET_CANDIDATES:
        print(f"       - {candidate}")
print(f"[PATH] Classical features: {CLASSICAL_FEATURES}")
print(f"[PATH] Artifacts:          {ARTIFACTS_DIR}")
print(f"[PATH] DFT logs:           {DFT_LOG_DIR}")
print(f"[PATH] xTB calculations:   {XTB_CALCULATIONS_DIR}")
print(f"[PATH] Plots:              {PLOTS_DIR}")

# DFT settings
DFT_FUNCTIONAL = "b3lyp"
DFT_BASIS      = "sto-3g"        # Fast smoke-test basis; switch to "6-31g*" for production
MAX_SCF_CYCLES = 50              # Fast smoke-test SCF limit; use 200 for production
SCF_CONV_TOL   = 1e-6            # Looser smoke-test SCF tolerance; use 1e-9 for production
GEOMOPT_MAX_STEPS = 3            # Fast smoke-test geometry steps; use 100 for production
GEOMOPT_GRAD_TOL = 4.5e-4        # Hartree/Bohr max-gradient target for Cartesian optimizer
TIMEOUT_PER_MOL = 600            # Informational runtime budget per molecule
USE_D3         = True            # DFT-D3 vdW correction
REQUIRE_D3     = True            # Stop early if D3 is unavailable
REQUIRE_GEOMOPT = True           # Stop early if the SciPy/PySCF optimizer is unavailable
ALLOW_SINGLE_POINT_FALLBACK = True  # Keep processing if a molecule-specific optimization fails
DFT_SOLVENT = None                  # Current PySCF workflow is gas phase; keep xTB gas phase too.

# Conformer generation
N_CONFORMERS   = 30
MMFF_MAX_ITERS = 500
RANDOM_SEED    = 42

# Test-run control. Default is one molecule for fast pipeline validation.
# Set P3Z_MAX_MOLECULES=0 for the full dataset after the smoke test passes.
MAX_DFT_MOLECULES = int(os.environ.get("P3Z_MAX_MOLECULES", "1"))

# Checkpoint: save progress every N molecules
CHECKPOINT_EVERY = 5

# DFT diagnostics. Concise optimization logs are enabled by default for smoke tests only.
DFT_DIAGNOSTICS = os.environ.get(
    "P3Z_DFT_DIAGNOSTICS",
    "1" if 0 < MAX_DFT_MOLECULES <= 10 else "0",
) == "1"
DFT_WRITE_LOGS = os.environ.get("P3Z_WRITE_DFT_LOGS", "1") == "1"
PYSCF_VERBOSE = int(os.environ.get("P3Z_PYSCF_VERBOSE", "0"))  # Notebook output verbosity.
PYSCF_LOG_VERBOSE = int(os.environ.get(
    "P3Z_PYSCF_LOG_VERBOSE",
    "4" if 0 < MAX_DFT_MOLECULES <= 10 else "3",
))  # File log verbosity; use 4 to inspect SCF cycles.
USE_XTB_PREOPT = os.environ.get("P3Z_USE_XTB_PREOPT", "1") == "1"
REQUIRE_XTB = os.environ.get("P3Z_REQUIRE_XTB", "1") == "1"
XTB_OPT_LEVEL = os.environ.get("P3Z_XTB_OPT_LEVEL", "tight")
XTB_MAX_CYCLES = int(os.environ.get("P3Z_XTB_MAX_CYCLES", "500"))
XTB_TIMEOUT_SECONDS = int(os.environ.get("P3Z_XTB_TIMEOUT_SECONDS", str(TIMEOUT_PER_MOL)))
XTB_OVERWRITE = os.environ.get("P3Z_XTB_OVERWRITE", "0") == "1"
XTB_DRY_RUN = os.environ.get("P3Z_XTB_DRY_RUN", "0") == "1"
AUTO_INSTALL_XTB_IN_COLAB = os.environ.get("P3Z_AUTO_INSTALL_XTB_IN_COLAB", "1") == "1"
ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE = os.environ.get(
    "P3Z_ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE",
    "0",
) == "1"
_requested_xtb_solvent = os.environ.get("P3Z_XTB_SOLVENT", "").strip()
if DFT_SOLVENT is None and _requested_xtb_solvent:
    print(f"[WARN] DFT is gas phase; ignoring P3Z_XTB_SOLVENT={_requested_xtb_solvent!r} for xTB consistency.")
    XTB_SOLVENT = None
else:
    XTB_SOLVENT = _requested_xtb_solvent or DFT_SOLVENT
RERUN_FAILED_CHECKPOINTS = os.environ.get(
    "P3Z_RERUN_FAILED_CHECKPOINTS",
    "1" if MAX_DFT_MOLECULES > 0 else "0",
) == "1"

if IS_COLAB:
    import psutil
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_count = os.cpu_count() or 2
    N_WORKERS = cpu_count
    PYSCF_MAX_MEMORY = 3000
    print(f"[COLAB] CPUs: {cpu_count} | Workers: {N_WORKERS} | RAM: {total_ram_gb:.1f} GB | PySCF mem/worker: {PYSCF_MAX_MEMORY} MB")
else:
    N_WORKERS = 1
    PYSCF_MAX_MEMORY = 1500
    print(f"[LOCAL] Workers: {N_WORKERS} | PySCF mem/worker: {PYSCF_MAX_MEMORY} MB")

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Checkpoint file (resumes from last successful molecule). Use a separate checkpoint for test runs.
if MAX_DFT_MOLECULES > 0:
    CHECKPOINT_FILE = ARTIFACTS_DIR / f"_dft_checkpoint_smoke_{MAX_DFT_MOLECULES}.parquet"
    DFT_RUN_LOG_FILE = DFT_LOG_DIR / f"dft_run_smoke_{MAX_DFT_MOLECULES}.log"
    print(f"[TEST] MAX_DFT_MOLECULES={MAX_DFT_MOLECULES}; set P3Z_MAX_MOLECULES=0 for the full dataset.")
else:
    CHECKPOINT_FILE = ARTIFACTS_DIR / "_dft_checkpoint.parquet"
    DFT_RUN_LOG_FILE = DFT_LOG_DIR / "dft_run_full.log"
    print("[FULL] MAX_DFT_MOLECULES=0; full labeled dataset will be processed.")
print(f"[DEBUG] DFT diagnostics: {DFT_DIAGNOSTICS} | notebook PySCF verbose: {PYSCF_VERBOSE}")
print(f"[DEBUG] DFT log files: {DFT_WRITE_LOGS} | PySCF file verbose: {PYSCF_LOG_VERBOSE}")
print(f"[DEBUG] xTB preoptimization: {USE_XTB_PREOPT} | opt={XTB_OPT_LEVEL} | cycles={XTB_MAX_CYCLES} | solvent={XTB_SOLVENT or 'gas'}")
print(f"[DEBUG] Auto-install xTB in Colab if missing: {AUTO_INSTALL_XTB_IN_COLAB}")
print(f"[DEBUG] xTB fallback to MMFF after failure: {ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE}")
print(f"[DEBUG] Rerun failed checkpoint rows: {RERUN_FAILED_CHECKPOINTS}")

# %% cell 6
# ================================================================
# S2 — Detect PySCF, SciPy Optimizer, and D3
# ================================================================
print("\n" + "=" * 60)
print("  P3Z — DFT+D3 GEOMETRY OPTIMIZATION + DESCRIPTORS (PySCF)")
print("=" * 60)
print(f"  Functional: {DFT_FUNCTIONAL.upper()}")
print(f"  Basis Set:  {DFT_BASIS}")
print(f"  SCF Cycles: {MAX_SCF_CYCLES}")
print(f"  Geom Steps: {GEOMOPT_MAX_STEPS}")

try:
    import pyscf
    from pyscf import gto, dft as pyscf_dft, scf
    from pyscf.dft import numint
    print(f"\n[OK] PySCF {pyscf.__version__} detected.")
except ImportError:
    print("\n[FATAL] PySCF is not installed!")
    print("        Install with: pip install pyscf h5py")
    raise SystemExit(1)

# Geometry optimization uses a SciPy Cartesian optimizer with PySCF(+D3) analytic gradients.
GEOMOPT_AVAILABLE = False
GEOMOPT_BACKEND = "none"
try:
    from scipy.optimize import minimize
    GEOMOPT_AVAILABLE = True
    GEOMOPT_BACKEND = "SciPy L-BFGS-B Cartesian gradients"
    print("[OK] Geometry optimizer available: SciPy L-BFGS-B + PySCF nuclear gradients.")
except ImportError as exc:
    print(f"[WARN] SciPy optimizer not available: {exc}")
    print("       Install with: pip install scipy")
    if REQUIRE_GEOMOPT:
        raise SystemExit(1)

# Check for D3 dispersion correction. Current dftd3 uses dftd3.pyscf.energy().
D3_AVAILABLE = False
if USE_D3:
    try:
        import dftd3
        import dftd3.pyscf as d3pyscf
        D3_AVAILABLE = True
        d3_version = getattr(dftd3, "__version__", "unknown")
        print(f"[OK] dftd3 {d3_version} detected — D3 energy and gradients enabled.")
    except ImportError as exc:
        print(f"[WARN] dftd3 not available: {exc}")
        print("       Install with: pip install dftd3")
        if REQUIRE_D3:
            raise SystemExit(1)
        USE_D3 = False

# Detect xTB command-line executable for GFN2-xTB pre-optimization.
def _install_xtb_with_apt_in_colab():
    if not IS_COLAB:
        return None
    print("[COLAB] Trying Ubuntu apt package 'xtb'.")
    try:
        subprocess.check_call(["apt-get", "-qq", "update"])
        subprocess.check_call(["apt-get", "-qq", "install", "-y", "xtb"])
    except Exception as exc:
        print(f"[WARN] Colab apt install for xTB failed: {type(exc).__name__}: {exc}")
        return None
    return shutil.which("xtb")


def _ensure_micromamba_in_colab():
    """Download a standalone micromamba binary without changing the notebook Python."""
    micromamba_bin = Path("/content/p3z_micromamba/bin/micromamba")
    if micromamba_bin.exists():
        return micromamba_bin

    micromamba_bin.parent.mkdir(parents=True, exist_ok=True)
    archive_path = Path("/content/p3z_micromamba/micromamba.tar.bz2")
    print("[COLAB] Downloading micromamba to install xTB from conda-forge.")
    urllib.request.urlretrieve("https://micro.mamba.pm/api/micromamba/linux-64/latest", archive_path)
    with tarfile.open(archive_path, "r:bz2") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith("bin/micromamba")), None)
        if member is None:
            raise RuntimeError("micromamba archive did not contain bin/micromamba")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError("could not extract micromamba binary")
        micromamba_bin.write_bytes(extracted.read())
    micromamba_bin.chmod(0o755)
    return micromamba_bin


def _install_xtb_with_micromamba_in_colab():
    if not IS_COLAB:
        return None
    env_prefix = Path("/content/p3z_xtb_env")
    xtb_path = env_prefix / "bin" / "xtb"
    if xtb_path.exists():
        os.environ["PATH"] = f"{xtb_path.parent}:{os.environ.get('PATH', '')}"
        return str(xtb_path)

    try:
        micromamba_bin = _ensure_micromamba_in_colab()
        print("[COLAB] Installing xTB executable from conda-forge with micromamba.")
        env = os.environ.copy()
        env.setdefault("MAMBA_ROOT_PREFIX", "/content/p3z_micromamba/root")
        subprocess.check_call([
            str(micromamba_bin),
            "create",
            "-y",
            "-p",
            str(env_prefix),
            "-c",
            "conda-forge",
            "xtb",
        ], env=env)
    except Exception as exc:
        print(f"[WARN] micromamba xTB install failed: {type(exc).__name__}: {exc}")
        return None

    if xtb_path.exists():
        os.environ["PATH"] = f"{xtb_path.parent}:{os.environ.get('PATH', '')}"
        return str(xtb_path)
    return shutil.which("xtb")


def _install_xtb_in_colab_if_missing():
    """Install the external xtb executable in Colab only when it is missing."""
    if not IS_COLAB or not AUTO_INSTALL_XTB_IN_COLAB:
        return None
    print("[COLAB] xTB executable not found on PATH; installing external xTB executable.")
    return _install_xtb_with_apt_in_colab() or _install_xtb_with_micromamba_in_colab()


XTB_EXECUTABLE = None
XTB_VERSION = "not_available"
XTB_VERSION_RAW = ""
if USE_XTB_PREOPT:
    XTB_EXECUTABLE = shutil.which("xtb")
    if XTB_EXECUTABLE is None:
        XTB_EXECUTABLE = _install_xtb_in_colab_if_missing()
    if XTB_EXECUTABLE is None:
        print("[FATAL] xTB executable not found on PATH; GFN2-xTB pre-optimization is required.")
        print("        Local install: conda install -c conda-forge xtb")
        print("        Colab automatic install tries apt first, then micromamba/conda-forge under /content/p3z_xtb_env.")
        print("        Colab manual quick test: !which xtb && !xtb --version")
        print("        To disable only for debugging: set P3Z_USE_XTB_PREOPT=0 before running the notebook.")
        if REQUIRE_XTB:
            raise SystemExit(1)
    else:
        try:
            xtb_version_proc = subprocess.run(
                [XTB_EXECUTABLE, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )
            XTB_VERSION_RAW = (xtb_version_proc.stdout or "") + (xtb_version_proc.stderr or "")
            if xtb_version_proc.returncode != 0:
                raise RuntimeError(XTB_VERSION_RAW.strip() or f"xtb --version returned {xtb_version_proc.returncode}")
            xtb_match = re.search(r"(?:xtb\s+version|version)\s+([0-9][^\s]*)", XTB_VERSION_RAW, flags=re.IGNORECASE)
            XTB_VERSION = xtb_match.group(1) if xtb_match else (XTB_VERSION_RAW.strip().splitlines()[0] if XTB_VERSION_RAW.strip() else "unknown")
            print(f"[OK] xTB detected: {XTB_EXECUTABLE} | version: {XTB_VERSION}")
            print("[OK] GFN2-xTB pre-optimization will run before every DFT geometry optimization.")
        except Exception as exc:
            print(f"[FATAL] Could not run xtb --version: {type(exc).__name__}: {exc}")
            if REQUIRE_XTB:
                raise SystemExit(1)
else:
    print("[WARN] GFN2-xTB pre-optimization disabled by P3Z_USE_XTB_PREOPT=0.")

if not INPUT_PARQUET.exists():
    checked_paths = "\n".join(f"  - {path}" for path in INPUT_PARQUET_CANDIDATES)
    raise FileNotFoundError(
        "Input parquet not found. Put curated_molecules_with_splits.parquet in one of these locations:\n"
        f"{checked_paths}\n"
        "Or set P3Z_INPUT_PARQUET to the exact parquet path."
    )

# %% cell 7
# ================================================================
# S3 — Load Data
# ================================================================
df = pd.read_parquet(INPUT_PARQUET)
df_pass = df[df["qc_status"] == "pass"].copy().reset_index(drop=True)
df_labeled_all = df_pass[df_pass["repellent_active"].notna()].copy().reset_index(drop=True)

if MAX_DFT_MOLECULES > 0:
    df_pos_test = df_labeled_all[df_labeled_all["repellent_active"] == 1]
    df_neg_test = df_labeled_all[df_labeled_all["repellent_active"] == 0]
    if len(df_pos_test) > 0 and len(df_neg_test) > 0 and MAX_DFT_MOLECULES >= 2:
        n_pos_test = min((MAX_DFT_MOLECULES + 1) // 2, len(df_pos_test))
        n_neg_test = min(MAX_DFT_MOLECULES - n_pos_test, len(df_neg_test))
        shortfall = MAX_DFT_MOLECULES - n_pos_test - n_neg_test
        if shortfall > 0:
            n_pos_test = min(n_pos_test + shortfall, len(df_pos_test))
        df_labeled = pd.concat([
            df_pos_test.head(n_pos_test),
            df_neg_test.head(n_neg_test),
        ], axis=0).reset_index(drop=True)
        print(f"\n[TEST] Using balanced smoke subset: {n_pos_test} positives + {n_neg_test} negatives from {len(df_labeled_all)} labeled molecules.")
    else:
        df_labeled = df_labeled_all.head(MAX_DFT_MOLECULES).copy().reset_index(drop=True)
        print(f"\n[TEST] Using first {len(df_labeled)} of {len(df_labeled_all)} labeled molecules.")
else:
    df_labeled = df_labeled_all.copy().reset_index(drop=True)

n_pos = (df_labeled["repellent_active"] == 1).sum()
n_neg = (df_labeled["repellent_active"] == 0).sum()

print(f"\n[OK] Loaded {len(df_pass)} QC-pass molecules total.")
print(f"[OK] Computing DFT descriptors for {len(df_labeled)} labeled molecules.")
print(f"     Positives: {n_pos}")
print(f"     Negatives: {n_neg}")

# Check for existing checkpoint
already_done_ids = set()
if CHECKPOINT_FILE.exists():
    df_checkpoint = pd.read_parquet(CHECKPOINT_FILE)
    already_done_ids = set(df_checkpoint["compound_id"].tolist())
    print(f"\n[OK] CHECKPOINT FOUND: {len(already_done_ids)} molecules already computed. Resuming...")

# %% cell 8
# ================================================================
# S4 — Conformer Generation (ETKDG + MMFF)
# ================================================================
print("\n" + "=" * 60)
print("  STEP 1: 3D CONFORMER GENERATION")
print("=" * 60)

import pickle
if MAX_DFT_MOLECULES > 0:
    CONFORMER_CACHE = ARTIFACTS_DIR / f"_conformers_cache_smoke_{MAX_DFT_MOLECULES}.pkl"
else:
    CONFORMER_CACHE = ARTIFACTS_DIR / "_conformers_cache.pkl"
SELECTED_COMPOUND_IDS = set(df_labeled["compound_id"].astype(str))


def generate_best_conformer(smiles, n_confs=30, max_iters=500, seed=42):
    """
    Generate 3D conformers with ETKDG, MMFF-optimize, and
    return the lowest-energy 3D Mol object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, 0

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0
    params.useSmallRingTorsions = True

    try:
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    except Exception:
        conf_ids = []

    if len(conf_ids) == 0:
        params.useRandomCoords = True
        try:
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        except Exception:
            pass
        if len(conf_ids) == 0:
            return None, None, 0

    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters, numThreads=0)
    except Exception:
        return mol, 0.0, len(conf_ids)

    best_conf_id = None
    best_energy = float("inf")
    for conf_id, (converged, energy) in enumerate(results):
        if energy < best_energy:
            best_energy = energy
            best_conf_id = conf_id

    if best_conf_id is None:
        return mol, 0.0, len(conf_ids)

    mol_best = Chem.RWMol(mol)
    all_conf_ids = [c.GetId() for c in mol_best.GetConformers()]
    for cid in all_conf_ids:
        if cid != best_conf_id:
            mol_best.RemoveConformer(cid)

    return mol_best, best_energy, len(conf_ids)


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

# %% cell 10
# ================================================================
# S4b — GFN2-xTB Geometry Pre-Optimization Helpers
# ================================================================

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


def safe_xtb_name(value):
    text = str(value or "molecule")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return safe[:120] or "molecule"


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
    executable = executable or XTB_EXECUTABLE
    xtb_version = xtb_version or XTB_VERSION
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

# %% cell 11
# ================================================================
# S5 — DFT+D3 Geometry Optimization and Descriptor Function
# ================================================================

_DFT_ACTIVE_LOG_HANDLE = None
_DFT_ACTIVE_LOG_PATH = None


def _safe_log_stem(value):
    text = str(value or "molecule")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return safe[:120] or "molecule"


def _dft_log_line(message):
    if not DFT_WRITE_LOGS:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    try:
        DFT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with DFT_RUN_LOG_FILE.open("a", encoding="utf-8") as run_handle:
            run_handle.write(line)
        if _DFT_ACTIVE_LOG_HANDLE is not None:
            _DFT_ACTIVE_LOG_HANDLE.write(line)
            _DFT_ACTIVE_LOG_HANDLE.flush()
    except Exception as log_exc:
        if DFT_DIAGNOSTICS:
            print(f"  [WARN] Could not write DFT log: {type(log_exc).__name__}: {log_exc}", flush=True)


def dft_debug(message):
    text = str(message).rstrip()
    if DFT_DIAGNOSTICS:
        print(text, flush=True)
    for line in (text.splitlines() or [""]):
        _dft_log_line(line)


def dft_status(message):
    text = str(message).rstrip()
    print(text, flush=True)
    for line in (text.splitlines() or [""]):
        _dft_log_line(line)


@contextmanager
def dft_molecule_log(compound_id, molecule_index=None, total_molecules=None):
    global _DFT_ACTIVE_LOG_HANDLE, _DFT_ACTIVE_LOG_PATH
    if not DFT_WRITE_LOGS:
        yield None
        return

    previous_handle = _DFT_ACTIVE_LOG_HANDLE
    previous_path = _DFT_ACTIVE_LOG_PATH
    log_path = DFT_LOG_DIR / f"{_safe_log_stem(compound_id)}.log"
    DFT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        _DFT_ACTIVE_LOG_HANDLE = handle
        _DFT_ACTIVE_LOG_PATH = log_path
        index_text = f" molecule_index={molecule_index}/{total_molecules}" if molecule_index is not None else ""
        _dft_log_line("=" * 72)
        _dft_log_line(f"BEGIN compound_id={compound_id}{index_text} method={method_label()}")
        _dft_log_line(f"settings functional={DFT_FUNCTIONAL} basis={DFT_BASIS} D3={USE_D3 and D3_AVAILABLE} geom_max_steps={GEOMOPT_MAX_STEPS} scf_max_cycles={MAX_SCF_CYCLES}")
        try:
            yield log_path
        finally:
            _dft_log_line(f"END compound_id={compound_id}")
            _DFT_ACTIVE_LOG_HANDLE = previous_handle
            _DFT_ACTIVE_LOG_PATH = previous_path


def active_pyscf_verbose():
    if DFT_WRITE_LOGS and _DFT_ACTIVE_LOG_HANDLE is not None:
        return max(PYSCF_VERBOSE, PYSCF_LOG_VERBOSE)
    return PYSCF_VERBOSE


def configure_pyscf_logging(obj):
    try:
        obj.verbose = active_pyscf_verbose()
    except Exception:
        pass
    if DFT_WRITE_LOGS and _DFT_ACTIVE_LOG_HANDLE is not None:
        try:
            obj.stdout = _DFT_ACTIVE_LOG_HANDLE
        except Exception:
            pass
    return obj


def method_label():
    suffix = "-D3" if USE_D3 and D3_AVAILABLE else ""
    return f"DFT-{DFT_FUNCTIONAL}/{DFT_BASIS}{suffix}"


def rdkit_charge_and_spin(mol_3d):
    """Infer PySCF charge and spin from the same RDKit logic used for xTB."""
    charge, spin, _multiplicity = infer_rdkit_charge_spin_multiplicity(mol_3d)
    return charge, spin


def build_pyscf_molecule(mol_3d):
    """Build a PySCF Mole from the RDKit conformer coordinates in angstrom."""
    conf = mol_3d.GetConformer()
    positions = conf.GetPositions()

    atom_lines = []
    for i in range(mol_3d.GetNumAtoms()):
        sym = mol_3d.GetAtomWithIdx(i).GetSymbol()
        x, y, z = positions[i]
        atom_lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")

    charge, spin = rdkit_charge_and_spin(mol_3d)
    dft_debug(f"  [DFT] build PySCF molecule: atoms={mol_3d.GetNumAtoms()} charge={charge} spin={spin}")
    mol_pyscf = gto.M(
        atom="; ".join(atom_lines),
        basis=DFT_BASIS,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        verbose=0 if _DFT_ACTIVE_LOG_HANDLE is not None else PYSCF_VERBOSE,
        max_memory=PYSCF_MAX_MEMORY,
    )
    return configure_pyscf_logging(mol_pyscf)


def build_mean_field(mol_pyscf):
    """Create a DFT mean-field object, wrapped with D3 when requested."""
    if mol_pyscf.spin == 0:
        mf = pyscf_dft.RKS(mol_pyscf)
    else:
        mf = pyscf_dft.UKS(mol_pyscf)

    mf.xc = DFT_FUNCTIONAL
    mf.max_cycle = MAX_SCF_CYCLES
    mf.conv_tol = SCF_CONV_TOL
    mf.max_memory = PYSCF_MAX_MEMORY
    mf.verbose = PYSCF_VERBOSE
    mf.grids.level = 3  # Medium DFT grid (balance speed/accuracy)
    mf = configure_pyscf_logging(mf)

    if USE_D3 and D3_AVAILABLE:
        mf = d3pyscf.energy(mf)
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


def run_final_single_point(mol_pyscf):
    """Run final DFT(+D3) single point at the supplied geometry."""
    mf = build_mean_field(mol_pyscf)
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
        f"basis={DFT_BASIS}, D3={USE_D3 and D3_AVAILABLE}, maxiter={GEOMOPT_MAX_STEPS}"
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

    result = minimize(
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


def compute_dft_descriptors(mol_3d, compound_id=None):
    """
    Perform DFT+D3 geometry optimization and compute electronic + geometric descriptors.

    The final electronic descriptors and geometric descriptors are computed at the optimized
    geometry. If a molecule-specific optimization fails and ALLOW_SINGLE_POINT_FALLBACK is True,
    the row is retained as a DFT+D3 single-point calculation with opt_status describing the failure.
    """
    try:
        initial_mol = build_pyscf_molecule(mol_3d)
        mol_for_final = initial_mol
        opt_status = "single_point"
        opt_converged = False
        opt_iterations = 0
        opt_evaluations = 0
        opt_max_gradient = np.nan

        dft_debug(f"  [DFT] [{compound_id or 'molecule'}] start")

        if GEOMOPT_AVAILABLE:
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
                        return {
                            "method": method_label(),
                            "status": f"failed: {opt_status}",
                            "opt_status": opt_status,
                            "opt_converged": False,
                            "opt_iterations": opt_iterations,
                            "opt_evaluations": opt_evaluations,
                            "opt_max_gradient_hartree_per_bohr": opt_max_gradient,
                            "dispersion_corrected": USE_D3 and D3_AVAILABLE,
                            "descriptor_status": "skipped: optimization not converged",
                        }
            except Exception as opt_error:
                opt_status = f"opt_failed: {str(opt_error)[:160]}"
                mol_for_final = initial_mol
                dft_debug(traceback.format_exc(limit=4).rstrip())
                if compound_id:
                    dft_status(f"  [WARN] [{compound_id}] Optimization failed; using final single-point fallback")
                if not ALLOW_SINGLE_POINT_FALLBACK:
                    return {
                        "method": method_label(),
                        "status": f"failed: {opt_status}",
                        "opt_status": opt_status,
                        "opt_converged": False,
                        "opt_iterations": opt_iterations,
                        "opt_evaluations": opt_evaluations,
                        "opt_max_gradient_hartree_per_bohr": opt_max_gradient,
                        "dispersion_corrected": USE_D3 and D3_AVAILABLE,
                        "descriptor_status": "skipped: optimization failed",
                    }

        dft_debug(f"  [DFT] [{compound_id or 'molecule'}] final single-point start")
        sp_start = time.time()
        mf, total_energy = run_final_single_point(mol_for_final)
        dft_debug(
            f"  [DFT] [{compound_id or 'molecule'}] final SCF done: "
            f"E={float(total_energy):.10f} Ha converged={getattr(mf, 'converged', False)} "
            f"time={time.time() - sp_start:.1f}s"
        )

        if not getattr(mf, "converged", False):
            return {
                "method": method_label(),
                "status": "scf_not_converged",
                "opt_status": opt_status,
                "opt_converged": opt_converged,
                "opt_iterations": opt_iterations,
                "opt_evaluations": opt_evaluations,
                "opt_max_gradient_hartree_per_bohr": opt_max_gradient,
                "dispersion_corrected": USE_D3 and D3_AVAILABLE,
                "descriptor_status": "skipped: scf_not_converged",
            }

        # --- Extract Kohn-Sham orbital energies ---
        mo_energies_hartree, mo_occ = flatten_mo_energies_and_occ(mf)
        mo_energies_ev = mo_energies_hartree * 27.2114  # Hartree -> eV

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
        symbols = np.array([], dtype=object)
        coords = np.empty((0, 3), dtype=np.float64)

        try:
            symbols, coords = extract_optimized_geometry(mol_3d, mol_for_final)

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
            "dispersion_corrected": USE_D3 and D3_AVAILABLE,
            "dispersion_model": "DFT-D3" if USE_D3 and D3_AVAILABLE else "none",
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

        return descriptors

    except Exception as exc:
        error_msg = str(exc)[:240]
        dft_debug(traceback.format_exc(limit=5).rstrip())
        if compound_id:
            dft_status(f"  [ERROR] [{compound_id}] FAILED: {error_msg}")
        return {
            "method": method_label(),
            "status": f"failed: {error_msg}",
            "opt_status": "error",
            "opt_converged": False,
            "dispersion_corrected": USE_D3 and D3_AVAILABLE,
            "descriptor_status": f"skipped: {error_msg}",
        }

# %% cell 12
# ================================================================
# S5b — Fast DFT+D3 Smoke Test (Water, STO-3G)
# ================================================================
# This catches NumPy/SciPy/PySCF/dftd3 incompatibilities before the expensive full loop.
print("\n" + "=" * 60)
print("  SMOKE TEST: PySCF + DFT-D3 single-point and gradient")
print("=" * 60)

_smoke_basis = DFT_BASIS
_smoke_grid_level = 3
try:
    DFT_BASIS = "sto-3g"
    mol_smoke = gto.M(
        atom="O 0.000000 0.000000 0.000000; H 0.000000 -0.757000 0.587000; H 0.000000 0.757000 0.587000",
        basis=DFT_BASIS,
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
finally:
    DFT_BASIS = _smoke_basis

# %% cell 13
# ================================================================
# S5c — Lightweight xTB Integration Validation (No DFT Campaign)
# ================================================================
print("\n" + "=" * 60)
print("  VALIDATION TESTS: GFN2-xTB pre-optimization integration")
print("=" * 60)

_xtb_validation_root = ARTIFACTS_DIR / "_xtb_validation"
_xtb_validation_root.mkdir(parents=True, exist_ok=True)

print("\n[TEST xTB-1] Executable and version detection...")
assert not USE_XTB_PREOPT or XTB_EXECUTABLE is not None, "xTB executable should be detected when xTB preopt is enabled"
assert not USE_XTB_PREOPT or XTB_VERSION not in (None, "", "not_available"), "xTB version should be recorded"
print(f"  xTB executable: {XTB_EXECUTABLE}")
print(f"  xTB version:    {XTB_VERSION}")

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
    executable=XTB_EXECUTABLE,
    xtb_version=XTB_VERSION,
    dry_run=True,
)
assert dry_result["status"] == "dry_run"
assert Path(dry_result["input_geometry_path"]).exists()
dry_symbols, _ = read_xyz(dry_result["input_geometry_path"])
assert dry_symbols == water_symbols
assert dry_result["command"][dry_result["command"].index("--chrg") + 1] == "0"
assert dry_result["command"][dry_result["command"].index("--uhf") + 1] == "0"
print("  dry-run XYZ and command validated")

if USE_XTB_PREOPT and XTB_EXECUTABLE is not None and not XTB_DRY_RUN:
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
        executable=XTB_EXECUTABLE,
        xtb_version=XTB_VERSION,
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
        executable=XTB_EXECUTABLE,
        xtb_version=XTB_VERSION,
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
    executable=XTB_EXECUTABLE,
    xtb_version=XTB_VERSION,
    dry_run=True,
)
assert not failure_result["converged"]
assert failure_result["status"] == "invalid_charge_or_multiplicity"
assert Path(failure_result["status_json_path"]).exists()
print("  invalid multiplicity failure recorded in xtb_status.json")

assert DFT_FUNCTIONAL == "b3lyp"
assert DFT_BASIS == "sto-3g"
assert GEOMOPT_MAX_STEPS == 3
assert USE_D3 is True
assert XTB_SOLVENT is None
print("\n[OK] xTB integration validation passed; fast smoke-test DFT settings are active and xTB is gas-phase GFN2 without external D3/D4.")
print("=" * 60)

# %% cell 15
# ================================================================
# S5a — Geometric Descriptor Functions
# ================================================================

from typing import Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

# Atomic masses (g/mol) from standard atomic weights
ATOMIC_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'P': 30.974, 'S': 32.06, 'Cl': 35.45,
    'Br': 79.904, 'I': 126.90, 'Si': 28.085, 'B': 10.81,
    'K': 39.098, 'Ca': 40.078, 'Na': 22.990, 'Mg': 24.305,
}


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


# Check for ArbAlign availability
ARBALIGN_AVAILABLE = False
try:
    from arbalign import ArbalignContext
    ARBALIGN_AVAILABLE = True
except ImportError:
    pass


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

# %% cell 16
# ================================================================
# S5b — Run GFN2-xTB Pre-Optimization, then DFT+D3 Optimization
# ================================================================
print("\n" + "=" * 60)
print("  STEP 2: GFN2-xTB PRE-OPTIMIZATION + DFT+D3 GEOMETRY OPTIMIZATION")
print("=" * 60)

checkpoint_records = []
if CHECKPOINT_FILE.exists():
    df_checkpoint = pd.read_parquet(CHECKPOINT_FILE)
    checkpoint_records = df_checkpoint.to_dict(orient="records")
    print(f"\n[OK] Loaded checkpoint with {len(checkpoint_records)} rows: {CHECKPOINT_FILE.name}")


def checkpoint_has_required_xtb(record):
    if not USE_XTB_PREOPT:
        return True
    source = str(record.get("dft_start_geometry_source", ""))
    if source == "GFN2-xTB":
        return True
    if ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE and source == "MMFF_FALLBACK_AFTER_XTB_FAILURE":
        return True
    return False


records_by_id = {}
n_checkpoint_success = 0
n_checkpoint_failed_skipped = 0
n_checkpoint_incompatible = 0
for record in checkpoint_records:
    if "compound_id" not in record or pd.isna(record["compound_id"]):
        continue
    status = str(record.get("status", ""))
    compatible = checkpoint_has_required_xtb(record)
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
        print(f"     Recomputing rows without required xTB provenance: {n_checkpoint_incompatible}")

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
                "dispersion_corrected": USE_D3 and D3_AVAILABLE,
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
                        executable=XTB_EXECUTABLE,
                        xtb_version=XTB_VERSION,
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
                            "dispersion_corrected": USE_D3 and D3_AVAILABLE,
                            "descriptor_status": f"skipped: xTB {xtb_result.get('status')}",
                            "dft_start_geometry_source": dft_start_geometry_source,
                        }
                        result.update(xtb_fields)
                        dft_status(f"  [ERROR] [{cid}] xTB failed: {xtb_result.get('status')} — {xtb_result.get('message')}")
                except Exception as xtb_exc:
                    xtb_fields = {
                        "xtb_method": XTB_METHOD,
                        "xtb_version": XTB_VERSION,
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
                            "dispersion_corrected": USE_D3 and D3_AVAILABLE,
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
                    "xtb_version": XTB_VERSION,
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
            "dispersion_corrected": USE_D3 and D3_AVAILABLE,
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
    "dft_ionization_potential_ev", "dft_electron_affinity_ev",
    "dft_chemical_hardness_ev", "dft_electronegativity_ev",
    "dft_electrophilicity_ev", "dft_n_electrons", "dft_n_basis_functions",
    "formal_charge", "spin",
    "d_av_A", "ECN", "Rg_A", "HCM", "HD_A", "dQ_A", "alignment_RMSD_A",
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
    df_classical_labeled = df_classical[df_classical["repellent_active"].notna()].copy()

    df_q_features = df_dft_final[["compound_id"] + dft_available].copy()
    df_combined = df_classical_labeled.merge(df_q_features, on="compound_id", how="left")

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
    "geometry_optimization": {
        "optimizer": GEOMOPT_BACKEND if GEOMOPT_AVAILABLE else "None (single-point only)",
        "enabled": GEOMOPT_AVAILABLE,
    },
    "dft_settings": {
        "functional": DFT_FUNCTIONAL,
        "basis_set": DFT_BASIS,
        "max_scf_cycles": MAX_SCF_CYCLES,
        "convergence_tolerance": SCF_CONV_TOL,
        "grid_level": 3,
        "dispersion_correction": "D3" if USE_D3 and D3_AVAILABLE else "None",
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
        "xtb_version": XTB_VERSION,
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
dft_method_final = f"{DFT_FUNCTIONAL.upper()}/{DFT_BASIS}" + ("-D3" if USE_D3 and D3_AVAILABLE else "")
opt_method = GEOMOPT_BACKEND if GEOMOPT_AVAILABLE else "Single-point (no optimizer)"

print("\n" + "=" * 60)
print("  P3Z COMPLETE — DFT Optimization + Geometric Descriptors")
print("=" * 60)
print(f"  Total Molecules:         {len(df_labeled)}")
print(f"  Conformer Success:       {n_conf_success}/{len(df_labeled)}")
print(f"  xTB Pre-Optimization:    {XTB_METHOD if USE_XTB_PREOPT else 'Disabled'} ({XTB_OPT_LEVEL}, {XTB_SOLVENT or 'gas'})")
print(f"  xTB Converged:           {n_xtb_converged}/{len(df_labeled)}")
print(f"  DFT Method:              {dft_method_final}")
print(f"  Geometry Optimization:   {opt_method}")
print(f"  Total Converged:         {n_converged}")
print(f"  Geometries Optimized:    {n_opt_success}")
print(f"  DFT Features:            {len(dft_available)}")
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
display_cols = ["compound_id", "xtb_converged", "dft_start_geometry_source", "d_av_A", "ECN", "Rg_A", "HCM", "descriptor_status"]
display_cols_available = [c for c in display_cols if c in descriptor_summary.columns]
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.6f}' if not np.isnan(x) else 'NaN')
print(descriptor_summary[display_cols_available].head(10).to_string())
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.float_format')

