from __future__ import annotations

from pathlib import Path

from p3z_dft.config import (
    ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE,
    COMPUTE_HD_SHIFTS,
    COMPUTE_IR_SPECTRA,
    COMPUTE_VERTICAL_EA,
    COMPUTE_VIBRATIONS,
    RUN_SETTINGS_FINGERPRINT,
    USE_XTB_PREOPT,
)

def checkpoint_has_required_xtb(record):
    if not USE_XTB_PREOPT:
        return True
    source = str(record.get("dft_start_geometry_source", ""))
    if source == "GFN2-xTB":
        return True
    if ALLOW_DFT_FROM_ORIGINAL_GEOMETRY_AFTER_XTB_FAILURE and source == "MMFF_FALLBACK_AFTER_XTB_FAILURE":
        return True
    return False


def checkpoint_matches_current_run(record):
    if str(record.get("run_settings_fingerprint", "")) != RUN_SETTINGS_FINGERPRINT:
        return False
    if not checkpoint_has_required_xtb(record):
        return False
    if COMPUTE_VIBRATIONS:
        if str(record.get("dft_vibrational_status", "")) != "success":
            return False
        if not record.get("vibration_artifact_dir") or not Path(str(record.get("vibration_artifact_dir"))).exists():
            return False
    if COMPUTE_IR_SPECTRA:
        if not record.get("ir_spectrum_artifact_path") or not Path(str(record.get("ir_spectrum_artifact_path"))).exists():
            return False
    if COMPUTE_HD_SHIFTS:
        hd_status = str(record.get("hd_shift_status", ""))
        if hd_status not in {"success", "skipped_no_hydrogen"}:
            return False
        if hd_status == "success" and (not record.get("hd_shift_artifact_dir") or not Path(str(record.get("hd_shift_artifact_dir"))).exists()):
            return False
    if COMPUTE_VERTICAL_EA:
        if str(record.get("dft_vertical_ea_status", "")) != "success":
            return False
        if not record.get("anion_artifact_path") or not Path(str(record.get("anion_artifact_path"))).exists():
            return False
    return True

