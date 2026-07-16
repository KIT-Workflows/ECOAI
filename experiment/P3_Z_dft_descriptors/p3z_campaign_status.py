#!/usr/bin/env python
"""Report shard/campaign progress for P3-Z SLURM runs (no pipeline import noise)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from p3z_dft.shard_progress import campaign_progress, shard_checkpoint_path, shard_progress


def _resolve_paths() -> tuple[Path, Path, int, str]:
    xeco_root = Path(os.environ.get("XECO_ROOT", "/home/ws/xd2484/xeco")).expanduser().resolve()
    script_dir = Path(
        os.environ.get("P3Z_SCRIPT_DIR", xeco_root / "experiment" / "P3_Z_dft_descriptors")
    ).expanduser().resolve()
    artifacts_dir = Path(
        os.environ.get("P3Z_ARTIFACTS_DIR", script_dir / "artifacts_prod_electronic_v1")
    ).expanduser().resolve()
    input_parquet = Path(
        os.environ.get(
            "P3Z_INPUT_PARQUET",
            xeco_root / "experiment" / "P2_scaffold_splitting" / "artifacts" / "curated_molecules_with_splits.parquet",
        )
    ).expanduser().resolve()
    num_shards = int(os.environ.get("P3Z_NUM_SHARDS", "8"))
    selection_mode = os.environ.get("P3Z_SELECTION_MODE", "all_qc_pass").strip().lower()
    return artifacts_dir, input_parquet, num_shards, selection_mode


def _load_selected(input_parquet: Path, selection_mode: str) -> pd.DataFrame:
    df = pd.read_parquet(input_parquet)
    df_pass = df[df["qc_status"] == "pass"].copy().reset_index(drop=True)
    if selection_mode == "labeled_only":
        return df_pass[df_pass["repellent_active"].notna()].copy().reset_index(drop=True)
    return df_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="P3-Z campaign progress")
    parser.add_argument("--shard", type=int, default=None, help="Report one shard index only")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    artifacts_dir, input_parquet, num_shards, selection_mode = _resolve_paths()
    df_selected = _load_selected(input_parquet, selection_mode)

    if args.shard is not None:
        if not (0 <= args.shard < num_shards):
            raise SystemExit(f"--shard must be in [0, {num_shards})")
        checkpoint_path = shard_checkpoint_path(artifacts_dir, args.shard, num_shards)
        expected, converged, remaining = shard_progress(df_selected, args.shard, num_shards, checkpoint_path)
        payload = {
            "shard_index": args.shard,
            "num_shards": num_shards,
            "expected": expected,
            "converged": converged,
            "remaining": remaining,
            "complete": remaining == 0,
            "checkpoint": str(checkpoint_path),
            "checkpoint_exists": checkpoint_path.exists(),
        }
        if args.json:
            print(json.dumps(payload))
        else:
            print(
                f"shard {args.shard + 1}/{num_shards}: "
                f"{converged}/{expected} converged ({remaining} remaining)"
            )
        return 0 if payload["complete"] else 1

    expected_total, converged_total, remaining_total = campaign_progress(
        df_selected,
        artifacts_dir,
        num_shards,
    )
    shard_rows = []
    for shard_index in range(num_shards):
        checkpoint_path = shard_checkpoint_path(artifacts_dir, shard_index, num_shards)
        expected, converged, remaining = shard_progress(
            df_selected,
            shard_index,
            num_shards,
            checkpoint_path,
        )
        shard_rows.append(
            {
                "shard_index": shard_index,
                "expected": expected,
                "converged": converged,
                "remaining": remaining,
                "complete": remaining == 0,
                "checkpoint": str(checkpoint_path),
                "checkpoint_exists": checkpoint_path.exists(),
            }
        )

    payload = {
        "num_shards": num_shards,
        "selection_mode": selection_mode,
        "expected_total": expected_total,
        "converged_total": converged_total,
        "remaining_total": remaining_total,
        "complete": remaining_total == 0,
        "artifacts_dir": str(artifacts_dir),
        "conformer_cache_exists": (artifacts_dir / "_conformers_cache.pkl").exists(),
        "shards": shard_rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        cache_note = "yes" if payload["conformer_cache_exists"] else "no"
        print(f"artifacts: {artifacts_dir}")
        print(f"conformer cache: {cache_note}")
        print(f"campaign: {converged_total}/{expected_total} converged ({remaining_total} remaining)")
        for row in shard_rows:
            ckpt = "checkpoint" if row["checkpoint_exists"] else "no ckpt yet"
            print(
                f"  shard {row['shard_index'] + 1:>2}/{num_shards}: "
                f"{row['converged']:>4}/{row['expected']:<4}  remaining={row['remaining']}  ({ckpt})"
            )
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
