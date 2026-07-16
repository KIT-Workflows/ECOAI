from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from p3z_dft.shard_progress import shard_checkpoint_path


def merge_shard_checkpoints(artifacts_dir: Path, num_shards: int) -> Path:
    """Merge per-shard checkpoint parquet files into one combined checkpoint."""
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for shard_index in range(num_shards):
        path = shard_checkpoint_path(artifacts_dir, shard_index, num_shards)
        if not path.exists():
            missing.append(path.name)
            continue
        frames.append(pd.read_parquet(path))

    if missing:
        raise FileNotFoundError(
            f"Missing shard checkpoint(s): {', '.join(missing)}. "
            "Wait for all shard jobs to finish or resume incomplete shards first."
        )
    if not frames:
        raise FileNotFoundError(f"No shard checkpoints found under {artifacts_dir}")

    merged = pd.concat(frames, ignore_index=True)
    if merged["compound_id"].duplicated().any():
        duplicated = merged.loc[merged["compound_id"].duplicated(), "compound_id"].astype(str).tolist()
        raise ValueError(f"Duplicate compound_id values in merged checkpoint: {duplicated[:10]}")

    output_path = artifacts_dir / "_dft_checkpoint.parquet"
    merged.to_parquet(output_path, index=False, engine="pyarrow")

    summary = {
        "num_shards": num_shards,
        "rows_merged": int(len(merged)),
        "converged": int((merged.get("status", pd.Series("", index=merged.index)).astype(str) == "converged").sum()),
        "output_checkpoint": str(output_path),
    }
    summary_path = artifacts_dir / "_dft_checkpoint_merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Merged {summary['rows_merged']} rows from {num_shards} shards -> {output_path.name}")
    print(f"[OK] Converged rows: {summary['converged']}")
    print(f"[OK] Merge summary: {summary_path.name}")
    return output_path


def main_merge() -> None:
    from p3z_dft import config

    num_shards = config.NUM_SHARDS
    if num_shards < 2:
        raise ValueError("P3Z_MERGE_SHARDS=1 requires P3Z_NUM_SHARDS >= 2")
    merge_shard_checkpoints(config.ARTIFACTS_DIR, num_shards)


if __name__ == "__main__":
    try:
        main_merge()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
