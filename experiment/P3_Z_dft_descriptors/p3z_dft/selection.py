from __future__ import annotations

import zlib

import pandas as pd


def compound_shard_index(compound_id: str, num_shards: int) -> int:
    """Stable shard assignment for a compound_id (CRC32 mod num_shards)."""
    if num_shards <= 1:
        return 0
    digest = zlib.crc32(str(compound_id).encode("utf-8")) & 0xFFFFFFFF
    return int(digest % num_shards)


def filter_dataframe_shard(df_source: pd.DataFrame, shard_index: int, num_shards: int) -> pd.DataFrame:
    """Return rows assigned to shard_index out of num_shards."""
    if num_shards <= 1:
        return df_source.copy().reset_index(drop=True)
    compound_ids = df_source["compound_id"].astype(str)
    mask = compound_ids.map(lambda cid: compound_shard_index(cid, num_shards) == shard_index)
    return df_source.loc[mask].copy().reset_index(drop=True)

def select_balanced_labeled_subset(df_source, max_molecules):
    df_pos = df_source[df_source["repellent_active"] == 1].copy()
    df_neg = df_source[df_source["repellent_active"] == 0].copy()
    if len(df_pos) > 0 and len(df_neg) > 0 and max_molecules >= 2:
        n_pos = min((max_molecules + 1) // 2, len(df_pos))
        n_neg = min(max_molecules - n_pos, len(df_neg))
        shortfall = max_molecules - n_pos - n_neg
        if shortfall > 0:
            n_pos = min(n_pos + shortfall, len(df_pos))
        return pd.concat([df_pos.head(n_pos), df_neg.head(n_neg)], axis=0).reset_index(drop=True)
    return df_source.head(max_molecules).copy().reset_index(drop=True)


def select_representative_pilot(df_source, pilot_size):
    ordered = df_source.sort_values("compound_id").reset_index(drop=True)
    if pilot_size <= 0 or len(ordered) <= pilot_size:
        return ordered.copy(), {}

    buckets = []
    repellent = ordered[ordered["repellent_active"] == 1].copy()
    non_repellent = ordered[ordered["repellent_active"] == 0].copy()
    insecticide = ordered[(ordered["repellent_active"].isna()) & (ordered["source_dataset"] == "insecticide")].copy()
    natural_product = ordered[(ordered["repellent_active"].isna()) & (ordered["source_dataset"] == "natural_product")].copy()
    for name, bucket in [
        ("repellent", repellent),
        ("non_repellent", non_repellent),
        ("insecticide", insecticide),
        ("natural_product", natural_product),
    ]:
        if len(bucket) > 0:
            buckets.append((name, bucket))

    selected_rows = []
    selected_ids = set()
    offsets = {name: 0 for name, _bucket in buckets}
    bucket_counts = {name: 0 for name, _bucket in buckets}

    while len(selected_rows) < pilot_size and buckets:
        progress = False
        for name, bucket in buckets:
            idx = offsets[name]
            if idx >= len(bucket):
                continue
            row = bucket.iloc[idx]
            offsets[name] = idx + 1
            cid = str(row["compound_id"])
            if cid in selected_ids:
                continue
            selected_rows.append(row)
            selected_ids.add(cid)
            bucket_counts[name] += 1
            progress = True
            if len(selected_rows) >= pilot_size:
                break
        if not progress:
            break

    if len(selected_rows) < pilot_size:
        for _, row in ordered.iterrows():
            cid = str(row["compound_id"])
            if cid in selected_ids:
                continue
            selected_rows.append(row)
            selected_ids.add(cid)
            if len(selected_rows) >= pilot_size:
                break

    return pd.DataFrame(selected_rows).reset_index(drop=True), bucket_counts

