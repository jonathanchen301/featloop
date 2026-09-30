#!/usr/bin/env python3
"""Batch-score Production model onto holdout window."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from featloop import config  # noqa: E402
from featloop.serve import batch_score  # noqa: E402


def main() -> None:
    preds = batch_score(stage="Production")
    print(f"Wrote {len(preds)} scores → {config.PREDICTIONS}")
    print(preds.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
