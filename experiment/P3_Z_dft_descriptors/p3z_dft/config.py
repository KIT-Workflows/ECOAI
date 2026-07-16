"""Path resolution, environment toggles, and run settings for P3-Z."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

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

# Test-run control. Default is one molecule for fast pipeline validation.
# Set P3Z_MAX_MOLECULES=0 for a production-profile run.
MAX_DFT_MOLECULES = int(os.environ.get("P3Z_MAX_MOLECULES", "1"))
IS_SMOKE_RUN = MAX_DFT_MOLECULES > 0


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


# DFT settings
DFT_FUNCTIONAL = os.environ.get("P3Z_DFT_FUNCTIONAL", "b3lyp")
DFT_BASIS = os.environ.get("P3Z_DFT_BASIS", "sto-3g" if IS_SMOKE_RUN else "6-31g*")
MAX_SCF_CYCLES = _env_int("P3Z_MAX_SCF_CYCLES", 50 if IS_SMOKE_RUN else 200)
SCF_CONV_TOL = _env_float("P3Z_SCF_CONV_TOL", 1e-6 if IS_SMOKE_RUN else 1e-9)
GEOMOPT_MAX_STEPS = _env_int("P3Z_GEOMOPT_MAX_STEPS", 3 if IS_SMOKE_RUN else 100)
GEOMOPT_GRAD_TOL = _env_float("P3Z_GEOMOPT_GRAD_TOL", 4.5e-4)
TIMEOUT_PER_MOL = _env_int("P3Z_TIMEOUT_PER_MOL", 600)
USE_D3 = _env_flag("P3Z_USE_D3", True)
REQUIRE_D3 = _env_flag("P3Z_REQUIRE_D3", True)
REQUIRE_GEOMOPT = _env_flag("P3Z_REQUIRE_GEOMOPT", True)
ALLOW_SINGLE_POINT_FALLBACK = _env_flag("P3Z_ALLOW_SINGLE_POINT_FALLBACK", True)
DFT_SOLVENT = None  # Current PySCF workflow is gas phase; keep xTB gas phase too.

# Conformer generation
N_CONFORMERS = _env_int("P3Z_N_CONFORMERS", 30)
MMFF_MAX_ITERS = _env_int("P3Z_MMFF_MAX_ITERS", 500)
RANDOM_SEED = _env_int("P3Z_RANDOM_SEED", 42)

SELECTION_MODE = os.environ.get("P3Z_SELECTION_MODE", "labeled_only").strip().lower()
if SELECTION_MODE not in {"labeled_only", "all_qc_pass"}:
    raise ValueError("P3Z_SELECTION_MODE must be one of: labeled_only, all_qc_pass")

pilot_default = "representative" if (IS_SMOKE_RUN and SELECTION_MODE == "all_qc_pass") else "none"
PILOT_MODE = os.environ.get("P3Z_PILOT_MODE", pilot_default).strip().lower()
if PILOT_MODE not in {"none", "representative"}:
    raise ValueError("P3Z_PILOT_MODE must be one of: none, representative")
PILOT_SIZE = _env_int("P3Z_PILOT_SIZE", 0)
ACTIVE_SELECTION_LIMIT = PILOT_SIZE if (PILOT_MODE == "representative" and PILOT_SIZE > 0) else MAX_DFT_MOLECULES

COMPUTE_VIBRATIONS = _env_flag("P3Z_COMPUTE_VIBRATIONS", not IS_SMOKE_RUN)
COMPUTE_IR_SPECTRA = _env_flag("P3Z_COMPUTE_IR_SPECTRA", COMPUTE_VIBRATIONS and not IS_SMOKE_RUN)
COMPUTE_HD_SHIFTS = _env_flag("P3Z_COMPUTE_HD_SHIFTS", COMPUTE_VIBRATIONS and not IS_SMOKE_RUN)
COMPUTE_VERTICAL_EA = _env_flag("P3Z_COMPUTE_VERTICAL_EA", not IS_SMOKE_RUN)
if not COMPUTE_VIBRATIONS:
    COMPUTE_IR_SPECTRA = False
    COMPUTE_HD_SHIFTS = False

IR_SPECTRUM_MIN_CM1 = _env_float("P3Z_IR_SPECTRUM_MIN_CM1", 0.0)
IR_SPECTRUM_MAX_CM1 = _env_float("P3Z_IR_SPECTRUM_MAX_CM1", 4000.0)
IR_SPECTRUM_STEP_CM1 = _env_float("P3Z_IR_SPECTRUM_STEP_CM1", 1.0)
IR_SPECTRUM_FWHM_CM1 = _env_float("P3Z_IR_SPECTRUM_FWHM_CM1", 20.0)

VIBRATION_SCF_MAX_CYCLES = _env_int("P3Z_VIBRATION_SCF_MAX_CYCLES", 120 if IS_SMOKE_RUN else 200)
VIBRATION_SCF_CONV_TOL = _env_float("P3Z_VIBRATION_SCF_CONV_TOL", 1e-8 if IS_SMOKE_RUN else 1e-9)
VIBRATION_GRID_LEVEL = _env_int("P3Z_VIBRATION_GRID_LEVEL", 3)
ANION_SCF_MAX_CYCLES = _env_int("P3Z_ANION_SCF_MAX_CYCLES", 120 if IS_SMOKE_RUN else 200)
ANION_SCF_CONV_TOL = _env_float("P3Z_ANION_SCF_CONV_TOL", 1e-8 if IS_SMOKE_RUN else 1e-9)
ANION_GRID_LEVEL = _env_int("P3Z_ANION_GRID_LEVEL", 3)

# Checkpoint: save progress every N molecules
CHECKPOINT_EVERY = _env_int("P3Z_CHECKPOINT_EVERY", 5)

# Sharded SLURM array runs: split QC-pass molecules across independent workers.
NUM_SHARDS = _env_int("P3Z_NUM_SHARDS", 1)
_shard_index_env = os.environ.get("P3Z_SHARD_INDEX")
if _shard_index_env is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
    SHARD_INDEX = int(os.environ["SLURM_ARRAY_TASK_ID"])
else:
    SHARD_INDEX = _env_int("P3Z_SHARD_INDEX", 0)
if NUM_SHARDS < 1:
    raise ValueError("P3Z_NUM_SHARDS must be >= 1")
if not (0 <= SHARD_INDEX < NUM_SHARDS):
    raise ValueError(f"P3Z_SHARD_INDEX must be in [0, {NUM_SHARDS})")

SKIP_VALIDATION = _env_flag("P3Z_SKIP_VALIDATION", NUM_SHARDS > 1)
PREPARE_CONFORMERS_ONLY = _env_flag("P3Z_PREPARE_CONFORMERS_ONLY", False)
AUTO_RESUBMIT_ON_INCOMPLETE = _env_flag("P3Z_AUTO_RESUBMIT_ON_INCOMPLETE", False)

# DFT diagnostics. Concise optimization logs are enabled by default for small runs only.
DFT_DIAGNOSTICS = _env_flag(
    "P3Z_DFT_DIAGNOSTICS",
    0 < ACTIVE_SELECTION_LIMIT <= 10,
)
DFT_WRITE_LOGS = _env_flag("P3Z_WRITE_DFT_LOGS", True)
PYSCF_VERBOSE = _env_int("P3Z_PYSCF_VERBOSE", 0)  # Notebook output verbosity.
PYSCF_LOG_VERBOSE = _env_int(
    "P3Z_PYSCF_LOG_VERBOSE",
    4 if 0 < ACTIVE_SELECTION_LIMIT <= 10 else 3,
)
USE_XTB_PREOPT = _env_flag("P3Z_USE_XTB_PREOPT", True)
REQUIRE_XTB = _env_flag("P3Z_REQUIRE_XTB", True)
XTB_OPT_LEVEL = os.environ.get("P3Z_XTB_OPT_LEVEL", "tight")
XTB_MAX_CYCLES = _env_int("P3Z_XTB_MAX_CYCLES", 500)
XTB_TIMEOUT_SECONDS = _env_int("P3Z_XTB_TIMEOUT_SECONDS", TIMEOUT_PER_MOL)
XTB_OVERWRITE = _env_flag("P3Z_XTB_OVERWRITE", False)
XTB_DRY_RUN = _env_flag("P3Z_XTB_DRY_RUN", False)
AUTO_INSTALL_XTB_IN_COLAB = _env_flag("P3Z_AUTO_INSTALL_XTB_IN_COLAB", True)
ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE = _env_flag(
    "P3Z_ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE",
    False,
)
_requested_xtb_solvent = os.environ.get("P3Z_XTB_SOLVENT", "").strip()
if DFT_SOLVENT is None and _requested_xtb_solvent:
    print(f"[WARN] DFT is gas phase; ignoring P3Z_XTB_SOLVENT={_requested_xtb_solvent!r} for xTB consistency.")
    XTB_SOLVENT = None
else:
    XTB_SOLVENT = _requested_xtb_solvent or DFT_SOLVENT
RERUN_FAILED_CHECKPOINTS = _env_flag(
    "P3Z_RERUN_FAILED_CHECKPOINTS",
    ACTIVE_SELECTION_LIMIT > 0,
)

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

RUN_SETTINGS = {
    "dft_functional": DFT_FUNCTIONAL,
    "dft_basis": DFT_BASIS,
    "max_scf_cycles": MAX_SCF_CYCLES,
    "scf_conv_tol": SCF_CONV_TOL,
    "geomopt_max_steps": GEOMOPT_MAX_STEPS,
    "geomopt_grad_tol": GEOMOPT_GRAD_TOL,
    "use_d3": USE_D3,
    "selection_mode": SELECTION_MODE,
    "compute_vibrations": COMPUTE_VIBRATIONS,
    "compute_ir_spectra": COMPUTE_IR_SPECTRA,
    "compute_hd_shifts": COMPUTE_HD_SHIFTS,
    "compute_vertical_ea": COMPUTE_VERTICAL_EA,
    "vibration_scf_max_cycles": VIBRATION_SCF_MAX_CYCLES,
    "vibration_scf_conv_tol": VIBRATION_SCF_CONV_TOL,
    "anion_scf_max_cycles": ANION_SCF_MAX_CYCLES,
    "anion_scf_conv_tol": ANION_SCF_CONV_TOL,
    "xtb_preopt": USE_XTB_PREOPT,
    "xtb_opt_level": XTB_OPT_LEVEL,
    "xtb_solvent": XTB_SOLVENT or "gas",
}
RUN_SETTINGS_FINGERPRINT = hashlib.sha256(
    json.dumps(RUN_SETTINGS, sort_keys=True).encode("utf-8")
).hexdigest()

# Checkpoint file (resumes from last successful molecule). Use separate files for smoke, pilot, and full runs.
if PILOT_MODE == "representative" and ACTIVE_SELECTION_LIMIT > 0:
    run_scope_tag = f"pilot_{ACTIVE_SELECTION_LIMIT}"
    CHECKPOINT_FILE = ARTIFACTS_DIR / f"_dft_checkpoint_{run_scope_tag}.parquet"
    DFT_RUN_LOG_FILE = DFT_LOG_DIR / f"dft_run_{run_scope_tag}.log"
    print(f"[PILOT] Representative pilot mode active with {ACTIVE_SELECTION_LIMIT} molecules.")
elif MAX_DFT_MOLECULES > 0:
    run_scope_tag = f"smoke_{MAX_DFT_MOLECULES}"
    CHECKPOINT_FILE = ARTIFACTS_DIR / f"_dft_checkpoint_{run_scope_tag}.parquet"
    DFT_RUN_LOG_FILE = DFT_LOG_DIR / f"dft_run_{run_scope_tag}.log"
    print(f"[TEST] MAX_DFT_MOLECULES={MAX_DFT_MOLECULES}; set P3Z_MAX_MOLECULES=0 for the production-profile dataset.")
elif NUM_SHARDS > 1:
    run_scope_tag = f"shard_{SHARD_INDEX:03d}_of_{NUM_SHARDS:03d}"
    CHECKPOINT_FILE = ARTIFACTS_DIR / f"_dft_checkpoint_{run_scope_tag}.parquet"
    DFT_RUN_LOG_FILE = DFT_LOG_DIR / f"dft_run_{run_scope_tag}.log"
    print(f"[SHARD] Shard {SHARD_INDEX + 1}/{NUM_SHARDS} for {SELECTION_MODE} selection.")
else:
    run_scope_tag = "full"
    CHECKPOINT_FILE = ARTIFACTS_DIR / "_dft_checkpoint.parquet"
    DFT_RUN_LOG_FILE = DFT_LOG_DIR / "dft_run_full.log"
    print(f"[FULL] Production-profile run for {SELECTION_MODE} selection.")
print(f"[DEBUG] DFT diagnostics: {DFT_DIAGNOSTICS} | notebook PySCF verbose: {PYSCF_VERBOSE}")
print(f"[DEBUG] DFT log files: {DFT_WRITE_LOGS} | PySCF file verbose: {PYSCF_LOG_VERBOSE}")
print(f"[DEBUG] xTB preoptimization: {USE_XTB_PREOPT} | opt={XTB_OPT_LEVEL} | cycles={XTB_MAX_CYCLES} | solvent={XTB_SOLVENT or 'gas'}")
print(f"[DEBUG] Auto-install xTB in Colab if missing: {AUTO_INSTALL_XTB_IN_COLAB}")
print(f"[DEBUG] xTB fallback to MMFF after failure: {ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE}")
print(f"[DEBUG] Shards: {SHARD_INDEX + 1}/{NUM_SHARDS} | skip_validation={SKIP_VALIDATION}")
print(f"[DEBUG] Selection mode: {SELECTION_MODE} | pilot mode: {PILOT_MODE} | limit={ACTIVE_SELECTION_LIMIT or 'all'}")
print(f"[DEBUG] Stages: vibrations={COMPUTE_VIBRATIONS} ir={COMPUTE_IR_SPECTRA} hd={COMPUTE_HD_SHIFTS} vertical_ea={COMPUTE_VERTICAL_EA}")
print(f"[DEBUG] Run fingerprint: {RUN_SETTINGS_FINGERPRINT[:12]}")
print(f"[DEBUG] Rerun failed checkpoint rows: {RERUN_FAILED_CHECKPOINTS}")
