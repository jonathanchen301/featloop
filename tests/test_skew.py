"""Skew detector smoke tests."""

from __future__ import annotations

import numpy as np

from featloop.monitor import population_stability_index, skew_report
import pandas as pd


def test_psi_identical_near_zero():
    x = np.random.default_rng(0).normal(size=2000)
    psi = population_stability_index(x, x.copy())
    assert psi < 0.05


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    expected = rng.normal(0, 1, size=2000)
    actual = rng.normal(2, 1, size=2000)
    psi = population_stability_index(expected, actual)
    assert psi > 0.25


def test_skew_report_keys():
    train = pd.DataFrame({"a": np.arange(100.0), "b": np.ones(100)})
    serve = pd.DataFrame({"a": np.arange(100.0) + 50, "b": np.ones(100)})
    report = skew_report(train, serve, columns=["a", "b"])
    assert set(report) == {"a", "b"}
    assert report["a"] > report["b"]
