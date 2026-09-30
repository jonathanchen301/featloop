"""Retrain gate must reject forced-bad canaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from featloop import config
from featloop.monitor import CanaryReport, run_canaries, should_promote


class _AlwaysHighModel:
    """Predicts extreme high demand for every row (destroys binary accuracy)."""

    def predict(self, X):
        return np.full(len(X), 1e9)


def test_should_promote_rejects_failed_canary():
    report = CanaryReport(
        passed=False,
        max_psi=0.5,
        psi_by_feature={"temp": 0.5},
        high_demand_accuracy=0.1,
        holdout_mae=999.0,
        reasons=["forced"],
    )
    ok, reasons = should_promote(report)
    assert ok is False
    assert reasons


def test_should_promote_rejects_worse_mae():
    report = CanaryReport(
        passed=True,
        max_psi=0.01,
        psi_by_feature={"temp": 0.01},
        high_demand_accuracy=0.9,
        holdout_mae=10.0,
        reasons=[],
    )
    ok, reasons = should_promote(report, challenger_mae=50.0, production_mae=10.0)
    assert ok is False
    assert any("MAE" in r for r in reasons)


def test_run_canaries_fails_on_bad_model():
    n = 400
    rng = np.random.default_rng(0)
    train = pd.DataFrame({c: rng.normal(size=n) for c in config.FEATURE_COLUMNS})
    # Mix of low/high labels so always-high preds cannot luck into accuracy
    train[config.LABEL_COLUMN] = rng.integers(1, 100, size=n).astype(float)
    serve = train.copy()
    # Force severe skew on a PSI-watched column
    serve["temp"] = serve["temp"] + 20.0
    report = run_canaries(train, serve, _AlwaysHighModel(), accuracy_min=0.55, psi_fail=5.0)
    assert report.passed is False
    assert (report.high_demand_accuracy < 0.55) or (report.max_psi >= 5.0)
