from __future__ import annotations

from pathlib import Path

import pandas as pd

from p3z_dft.selection import compound_shard_index, filter_dataframe_shard


def shard_checkpoint_path(artifacts_dir: Path, shard_index: int, num_shards: int) -> Path:
    return artifacts_dir / f"_dft_checkpoint_shard_{shard_index:03d}_of_{num_shards:03d}.parquet"


def count_converged_in_checkpoint(checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        return 0
    df = pd.read_parquet(checkpoint_path)
    status = df.get("status", pd.Series("", index=df.index)).fillna("").astype(str)
    return int((status == "converged").sum())


def shard_progress(
    df_selected: pd.DataFrame,
    shard_index: int,
    num_shards: int,
    checkpoint_path: Path,
) -> tuple[int, int, int]:
    """Return (expected, converged, remaining) for one shard."""
    shard_df = filter_dataframe_shard(df_selected, shard_index, num_shards)
    expected = len(shard_df)
    converged = count_converged_in_checkpoint(checkpoint_path)
    remaining = max(expected - converged, 0)
    return expected, converged, remaining


def campaign_progress(
    df_selected: pd.DataFrame,
    artifacts_dir: Path,
    num_shards: int,
) -> tuple[int, int, int]:
    """Return (expected_total, converged_total, remaining_total) across all shards."""
    expected_total = 0
    converged_total = 0
    for shard_index in range(num_shards):
        checkpoint_path = shard_checkpoint_path(artifacts_dir, shard_index, num_shards)
        expected, converged, _remaining = shard_progress(
            df_selected,
            shard_index,
            num_shards,
            checkpoint_path,
        )
        expected_total += expected
        converged_total += converged
    remaining_total = max(expected_total - converged_total, 0)
    return expected_total, converged_total, remaining_total


def shard_compound_ids(df_selected: pd.DataFrame, shard_index: int, num_shards: int) -> list[str]:
    compound_ids = df_selected["compound_id"].astype(str).tolist()
    return [cid for cid in compound_ids if compound_shard_index(cid, num_shards) == shard_index]
