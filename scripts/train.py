#!/usr/bin/env python3
"""Train baseline and register to Staging (then promote on demo)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from featloop.train import train_model  # noqa: E402


def main() -> None:
    result = train_model(register=True, stage_after="Staging", run_name="train")
    print(
        f"Train OK run_id={result.run_id} MAE={result.mae:.3f} RMSE={result.rmse:.3f} "
        f"n_train={result.n_train} n_valid={result.n_valid}"
    )
    print(f"model_uri={result.model_uri}")


if __name__ == "__main__":
    main()
