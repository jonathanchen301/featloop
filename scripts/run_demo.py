#!/usr/bin/env python3
"""One-shot stranger path: materialize → train → Production → score → retrain → canary."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from featloop import config  # noqa: E402
from featloop.features import materialize  # noqa: E402
from featloop.monitor import run_canaries, write_canary_artifacts  # noqa: E402
from featloop.registry import list_versions, load_stage, promote_to_production, transition_latest  # noqa: E402
from featloop.serve import batch_score  # noqa: E402
from featloop.train import train_model  # noqa: E402
from featloop.triggers import ensure_trigger  # noqa: E402
import pandas as pd  # noqa: E402
from featloop.features import time_split  # noqa: E402
from featloop.monitor import should_promote  # noqa: E402
from featloop.train import evaluate_mae  # noqa: E402
from featloop.registry import latest_version  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    print("=== FeatLoop demo ===")
    print(f"Root: {config.ROOT}")

    print("\n[1/5] Materialize offline feature table…")
    table = materialize()
    print(f"  → {config.FEATURE_TABLE} ({len(table)} rows)")

    print("\n[2/5] Train + register → Staging → Production…")
    r1 = train_model(register=True, stage_after="Staging", run_name="demo-baseline")
    v1 = latest_version()
    assert v1 is not None
    promote_to_production(v1)
    print(f"  → run={r1.run_id} MAE={r1.mae:.3f} Production=v{v1}")

    print("\n[3/5] Batch score Production…")
    preds = batch_score(stage="Production")
    print(f"  → {config.PREDICTIONS} ({len(preds)} rows)")

    print("\n[4/5] Event trigger + retrain challenger…")
    ensure_trigger(reason="demo_flash_event")
    r2 = train_model(register=True, stage_after="Staging", run_name="demo-challenger")
    v2 = latest_version()
    train, _, holdout = time_split(table)
    if config.TRAIN_SNAPSHOT.exists():
        train = pd.read_parquet(config.TRAIN_SNAPSHOT)
    challenger = load_stage("Staging")
    report = run_canaries(train, holdout, challenger)
    png = write_canary_artifacts(report)
    challenger_mae = evaluate_mae(challenger, holdout)
    prod_mae = evaluate_mae(load_stage("Production"), holdout)
    ok, reasons = should_promote(
        report, challenger_mae=challenger_mae, production_mae=prod_mae
    )
    if ok and v2 is not None:
        promote_to_production(v2)
        print(
            f"  → challenger v{v2} PROMOTED "
            f"(holdout MAE {challenger_mae:.3f} vs prod {prod_mae:.3f})"
        )
    else:
        print(f"  → challenger v{v2} kept in Staging (gate reject)")
        for reason in reasons:
            print(f"     - {reason}")
        transition_latest("Staging")

    print("\n[5/5] Final canary on Production…")
    prod = load_stage("Production")
    final = run_canaries(train, holdout, prod)
    write_canary_artifacts(final)

    elapsed = time.perf_counter() - t0
    versions = list_versions()
    print("\n=== Demo complete ===")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Feature table: {config.FEATURE_TABLE}")
    print(f"Predictions:   {config.PREDICTIONS}")
    print(f"Canary plot:   {png}")
    print(f"Canary JSON:   {config.REPORTS / 'canary.json'}")
    print(f"MLflow UI:     mlflow ui --backend-store-uri {config.MLRUNS}")
    print("Registry:")
    for v in versions:
        print(f"  {config.REGISTERED_MODEL_NAME} v{v['version']} [{v['stage']}]")
    if len(versions) < 2:
        raise SystemExit("Expected ≥2 model versions after retrain")
    print("\nHonesty note: metrics above are on the public UCI bike-share extract only —")
    print("not Guardians/AFRL production numbers.")


if __name__ == "__main__":
    main()
