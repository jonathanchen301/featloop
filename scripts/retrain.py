#!/usr/bin/env python3
"""Event-triggered retrain: challenger vs Production; promote only if canaries pass."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from featloop import config  # noqa: E402
from featloop.features import materialize, time_split  # noqa: E402
from featloop.monitor import run_canaries, should_promote, write_canary_artifacts  # noqa: E402
from featloop.registry import (  # noqa: E402
    latest_version,
    list_versions,
    load_stage,
    promote_to_production,
    transition_latest,
)
from featloop.train import evaluate_mae, train_model  # noqa: E402
from featloop.triggers import ensure_trigger, trigger_present  # noqa: E402


def main() -> None:
    if not trigger_present():
        ensure_trigger(reason="make_retrain")
        print(f"Created trigger {config.RETRAIN_TRIGGER}")

    print("Training challenger…")
    result = train_model(register=True, stage_after="Staging", run_name="retrain-challenger")
    challenger_version = latest_version()
    print(f"Challenger version={challenger_version} valid_MAE={result.mae:.3f}")

    table = materialize()
    train, _, holdout = time_split(table)
    if config.TRAIN_SNAPSHOT.exists():
        train = pd.read_parquet(config.TRAIN_SNAPSHOT)

    challenger = load_stage("Staging")
    report = run_canaries(train, holdout, challenger)
    write_canary_artifacts(report)

    challenger_mae = evaluate_mae(challenger, holdout)
    prod_mae = None
    try:
        prod = load_stage("Production")
        prod_mae = evaluate_mae(prod, holdout)
    except Exception:  # noqa: BLE001 — first run may lack Production
        prod_mae = None

    ok, reasons = should_promote(
        report, challenger_mae=challenger_mae, production_mae=prod_mae
    )
    if ok and challenger_version is not None:
        promote_to_production(challenger_version)
        print(f"PROMOTED version {challenger_version} → Production")
    else:
        print("REJECTED promote (fail-closed)")
        for r in reasons:
            print(f"  - {r}")
        # Leave challenger in Staging
        if challenger_version is not None:
            transition_latest("Staging")

    print("Registry versions:")
    for v in list_versions():
        print(f"  v{v['version']} stage={v['stage']}")


if __name__ == "__main__":
    main()
