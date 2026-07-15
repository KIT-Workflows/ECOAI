"""PySCF, D3, geometry optimizer, and xTB runtime detection."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import p3z_dft.config as cfg
from p3z_dft.config import (
    AUTO_INSTALL_XTB_IN_COLAB,
    DFT_BASIS,
    DFT_FUNCTIONAL,
    GEOMOPT_MAX_STEPS,
    INPUT_PARQUET,
    INPUT_PARQUET_CANDIDATES,
    IS_COLAB,
    MAX_SCF_CYCLES,
    REQUIRE_D3,
    REQUIRE_GEOMOPT,
    REQUIRE_XTB,
    USE_D3,
    USE_XTB_PREOPT,
)

# Populated by initialize_runtime().
pyscf = None
gto = None
pyscf_dft = None
scf = None
lib = None
numint = None
thermo = None
d3pyscf = None
minimize = None

GEOMOPT_AVAILABLE = False
GEOMOPT_BACKEND = "none"
D3_AVAILABLE = False

XTB_EXECUTABLE: str | None = None
XTB_VERSION = "not_available"
XTB_VERSION_RAW = ""


def _install_xtb_with_apt_in_colab() -> str | None:
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


def _ensure_micromamba_in_colab() -> Path:
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


def _install_xtb_with_micromamba_in_colab() -> str | None:
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
        subprocess.check_call(
            [
                str(micromamba_bin),
                "create",
                "-y",
                "-p",
                str(env_prefix),
                "-c",
                "conda-forge",
                "xtb",
            ],
            env=env,
        )
    except Exception as exc:
        print(f"[WARN] micromamba xTB install failed: {type(exc).__name__}: {exc}")
        return None

    if xtb_path.exists():
        os.environ["PATH"] = f"{xtb_path.parent}:{os.environ.get('PATH', '')}"
        return str(xtb_path)
    return shutil.which("xtb")


def _install_xtb_in_colab_if_missing() -> str | None:
    if not IS_COLAB or not AUTO_INSTALL_XTB_IN_COLAB:
        return None
    print("[COLAB] xTB executable not found on PATH; installing external xTB executable.")
    return _install_xtb_with_apt_in_colab() or _install_xtb_with_micromamba_in_colab()


def initialize_runtime(*, require_input: bool = True) -> None:
    """Detect PySCF, SciPy optimizer, D3, and xTB; optionally validate input parquet."""
    global pyscf, gto, pyscf_dft, scf, lib, numint, thermo
    global d3pyscf, minimize
    global GEOMOPT_AVAILABLE, GEOMOPT_BACKEND, D3_AVAILABLE
    global XTB_EXECUTABLE, XTB_VERSION, XTB_VERSION_RAW

    print("\n" + "=" * 60)
    print("  P3Z — DFT+D3 GEOMETRY OPTIMIZATION + DESCRIPTORS (PySCF)")
    print("=" * 60)
    print(f"  Functional: {DFT_FUNCTIONAL.upper()}")
    print(f"  Basis Set:  {DFT_BASIS}")
    print(f"  SCF Cycles: {MAX_SCF_CYCLES}")
    print(f"  Geom Steps: {GEOMOPT_MAX_STEPS}")

    try:
        import pyscf as _pyscf
        from pyscf import dft as _pyscf_dft
        from pyscf import gto as _gto
        from pyscf import lib as _lib
        from pyscf import scf as _scf
        from pyscf.dft import numint as _numint
        from pyscf.hessian import thermo as _thermo

        pyscf = _pyscf
        gto = _gto
        pyscf_dft = _pyscf_dft
        scf = _scf
        lib = _lib
        numint = _numint
        thermo = _thermo
        print(f"\n[OK] PySCF {pyscf.__version__} detected.")
    except ImportError:
        print("\n[FATAL] PySCF is not installed!")
        print("        Install with: pip install pyscf h5py")
        raise SystemExit(1) from None

    GEOMOPT_AVAILABLE = False
    GEOMOPT_BACKEND = "none"
    try:
        from scipy.optimize import minimize as _minimize

        minimize = _minimize
        GEOMOPT_AVAILABLE = True
        GEOMOPT_BACKEND = "SciPy L-BFGS-B Cartesian gradients"
        print("[OK] Geometry optimizer available: SciPy L-BFGS-B + PySCF nuclear gradients.")
    except ImportError as exc:
        print(f"[WARN] SciPy optimizer not available: {exc}")
        print("       Install with: pip install scipy")
        if REQUIRE_GEOMOPT:
            raise SystemExit(1) from exc

    D3_AVAILABLE = False
    if USE_D3:
        try:
            import dftd3
            import dftd3.pyscf as _d3pyscf

            d3pyscf = _d3pyscf
            D3_AVAILABLE = True
            d3_version = getattr(dftd3, "__version__", "unknown")
            print(f"[OK] dftd3 {d3_version} detected — D3 energy and gradients enabled.")
        except ImportError as exc:
            print(f"[WARN] dftd3 not available: {exc}")
            print("       Install with: pip install dftd3")
            if REQUIRE_D3:
                raise SystemExit(1) from exc
            cfg.USE_D3 = False

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
                    raise RuntimeError(
                        XTB_VERSION_RAW.strip() or f"xtb --version returned {xtb_version_proc.returncode}"
                    )
                xtb_match = re.search(
                    r"(?:xtb\s+version|version)\s+([0-9][^\s]*)",
                    XTB_VERSION_RAW,
                    flags=re.IGNORECASE,
                )
                XTB_VERSION = (
                    xtb_match.group(1)
                    if xtb_match
                    else (XTB_VERSION_RAW.strip().splitlines()[0] if XTB_VERSION_RAW.strip() else "unknown")
                )
                print(f"[OK] xTB detected: {XTB_EXECUTABLE} | version: {XTB_VERSION}")
                print("[OK] GFN2-xTB pre-optimization will run before every DFT geometry optimization.")
            except Exception as exc:
                print(f"[FATAL] Could not run xtb --version: {type(exc).__name__}: {exc}")
                if REQUIRE_XTB:
                    raise SystemExit(1) from exc
    else:
        print("[WARN] GFN2-xTB pre-optimization disabled by P3Z_USE_XTB_PREOPT=0.")

    if require_input and not INPUT_PARQUET.exists():
        checked_paths = "\n".join(f"  - {path}" for path in INPUT_PARQUET_CANDIDATES)
        raise FileNotFoundError(
            "Input parquet not found. Put curated_molecules_with_splits.parquet in one of these locations:\n"
            f"{checked_paths}\n"
            "Or set P3Z_INPUT_PARQUET to the exact parquet path."
        )
