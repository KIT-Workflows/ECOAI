"""CLI entry point for the P3-Z DFT pipeline."""

from __future__ import annotations

import os
import sys

from p3z_dft.merge import main_merge
from p3z_dft.pipeline import main


if __name__ == "__main__":
    if os.environ.get("P3Z_MERGE_SHARDS", "").strip().lower() in {"1", "true", "yes", "on"}:
        main_merge()
    else:
        main()
