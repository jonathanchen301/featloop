#!/usr/bin/env python3
"""Run skew + accuracy canaries against Production."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from featloop import config  # noqa: E402
from featloop.features import materialize, time_split  # noqa: E402
from featloop.monitor import run_canaries, write_canary_artifacts  # noqa: E402
from featloop.registry import load_stage  # noqa: E402


def main() -> None:
    table = materialize()
    train, _, holdout = time_split(table)
    if config.TRAIN_SNAPSHOT.exists():
        train = pd.read_parquet(config.TRAIN_SNAPSHOT)
    model = load_stage("Production")
    report = run_canaries(train, holdout, model)
    png = write_canary_artifacts(report)
    status = "PASS" if report.passed else "FAIL"
    print(f"Canary {status}: max_psi={report.max_psi:.3f} acc={report.high_demand_accuracy:.3f}")
    if report.reasons:
        for r in report.reasons:
            print(f"  - {r}")
    print(f"Report: {config.REPORTS / 'canary.json'}")
    print(f"Plot:   {png}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
