#!/usr/bin/env python
"""Report shard/campaign progress for P3-Z SLURM runs."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from p3z_dft import config
from p3z_dft.shard_progress import campaign_progress, shard_checkpoint_path, shard_progress


def _load_selected() -> pd.DataFrame:
    df = pd.read_parquet(config.INPUT_PARQUET)
    df_pass = df[df["qc_status"] == "pass"].copy().reset_index(drop=True)
    if config.SELECTION_MODE == "labeled_only":
        return df_pass[df_pass["repellent_active"].notna()].copy().reset_index(drop=True)
    return df_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="P3-Z campaign progress")
    parser.add_argument("--shard", type=int, default=None, help="Report one shard index only")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    df_selected = _load_selected()
    num_shards = config.NUM_SHARDS
    artifacts_dir = config.ARTIFACTS_DIR

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
            }
        )

    payload = {
        "num_shards": num_shards,
        "expected_total": expected_total,
        "converged_total": converged_total,
        "remaining_total": remaining_total,
        "complete": remaining_total == 0,
        "shards": shard_rows,
        "artifacts_dir": str(artifacts_dir),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"campaign: {converged_total}/{expected_total} converged ({remaining_total} remaining)")
        for row in shard_rows:
            print(
                f"  shard {row['shard_index'] + 1:>2}/{num_shards}: "
                f"{row['converged']:>4}/{row['expected']:<4}  remaining={row['remaining']}"
            )
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
