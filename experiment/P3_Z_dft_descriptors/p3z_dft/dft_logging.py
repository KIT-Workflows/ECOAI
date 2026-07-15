from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from p3z_dft import config
from p3z_dft.config import (
    DFT_DIAGNOSTICS,
    DFT_FUNCTIONAL,
    DFT_BASIS,
    DFT_LOG_DIR,
    DFT_RUN_LOG_FILE,
    DFT_WRITE_LOGS,
    GEOMOPT_MAX_STEPS,
    MAX_SCF_CYCLES,
    PYSCF_LOG_VERBOSE,
    PYSCF_VERBOSE,
)
from p3z_dft import runtime

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
        _dft_log_line(f"settings functional={DFT_FUNCTIONAL} basis={DFT_BASIS} D3={config.USE_D3 and runtime.D3_AVAILABLE} geom_max_steps={GEOMOPT_MAX_STEPS} scf_max_cycles={MAX_SCF_CYCLES}")
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
    suffix = "-D3" if config.USE_D3 and runtime.D3_AVAILABLE else ""
    return f"DFT-{DFT_FUNCTIONAL}/{DFT_BASIS}{suffix}"

