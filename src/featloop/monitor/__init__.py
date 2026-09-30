"""Skew (PSI) + accuracy canaries; promote gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

from featloop import config
from featloop.features import feature_matrix


@dataclass
class CanaryReport:
    passed: bool
    max_psi: float
    psi_by_feature: dict[str, float]
    high_demand_accuracy: float
    holdout_mae: float
    reasons: list[str]


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
    eps: float = 1e-4,
) -> float:
    """PSI between two 1-d distributions (equal-width bins on expected)."""
    expected = expected.astype(float)
    actual = actual.astype(float)
    # Guard constant columns
    if np.allclose(expected.min(), expected.max()):
        return 0.0
    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 2:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=cuts)
    act_counts, _ = np.histogram(actual, bins=cuts)
    exp_perc = exp_counts / max(exp_counts.sum(), 1) + eps
    act_perc = act_counts / max(act_counts.sum(), 1) + eps
    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def skew_report(
    train_df: pd.DataFrame,
    serve_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    columns = columns or config.PSI_FEATURE_COLUMNS
    return {
        col: population_stability_index(
            train_df[col].to_numpy(),
            serve_df[col].to_numpy(),
        )
        for col in columns
        if col in train_df.columns and col in serve_df.columns
    }


def high_demand_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """Binary canary: predict whether demand exceeds threshold (quantile cut)."""
    true_high = (y_true >= threshold).astype(int)
    pred_high = (y_pred >= threshold).astype(int)
    return float(accuracy_score(true_high, pred_high))


def run_canaries(
    train_df: pd.DataFrame,
    serve_df: pd.DataFrame,
    model,
    *,
    psi_fail: float = config.PSI_FAIL,
    accuracy_min: float = config.ACCURACY_MIN,
) -> CanaryReport:
    # Compare serve to the *recent* train window so calendar drift across years
    # does not look like a feature-recipe bug.
    ref_n = max(int(len(train_df) * 0.25), 200)
    ref = train_df.iloc[-ref_n:]
    psi = skew_report(ref, serve_df)
    max_psi = max(psi.values()) if psi else 0.0

    X, y = feature_matrix(serve_df)
    preds = model.predict(X)
    thr = float(np.quantile(train_df[config.LABEL_COLUMN], config.HIGH_DEMAND_QUANTILE))
    acc = high_demand_accuracy(y, preds, thr)
    mae = float(mean_absolute_error(y, preds))

    reasons: list[str] = []
    if max_psi >= psi_fail:
        reasons.append(f"max PSI {max_psi:.3f} >= fail threshold {psi_fail}")
    if acc < accuracy_min:
        reasons.append(f"high-demand accuracy {acc:.3f} < min {accuracy_min}")

    return CanaryReport(
        passed=len(reasons) == 0,
        max_psi=max_psi,
        psi_by_feature=psi,
        high_demand_accuracy=acc,
        holdout_mae=mae,
        reasons=reasons,
    )


def write_canary_artifacts(report: CanaryReport, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "canary.json"
    json_path.write_text(json.dumps(asdict(report), indent=2))

    # Plot top PSI features
    items = sorted(report.psi_by_feature.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], vals[::-1], color="#2c5f7c")
    ax.axvline(config.PSI_WARN, color="#c4a35a", linestyle="--", label=f"warn {config.PSI_WARN}")
    ax.axvline(config.PSI_FAIL, color="#a33b3b", linestyle="--", label=f"fail {config.PSI_FAIL}")
    ax.set_xlabel("PSI (train vs serve)")
    ax.set_title(
        f"FeatLoop canary — {'PASS' if report.passed else 'FAIL'} "
        f"(acc={report.high_demand_accuracy:.2f}, MAE={report.holdout_mae:.1f})"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    png_path = out_dir / "canary.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path


def should_promote(
    report: CanaryReport,
    challenger_mae: float | None = None,
    production_mae: float | None = None,
    worsen_ratio: float = config.MAE_WORSEN_RATIO,
) -> tuple[bool, list[str]]:
    """Fail-closed promote gate used by retrain."""
    reasons = list(report.reasons)
    if not report.passed:
        return False, reasons
    if (
        challenger_mae is not None
        and production_mae is not None
        and challenger_mae > production_mae * worsen_ratio
    ):
        reasons.append(
            f"challenger MAE {challenger_mae:.2f} > prod {production_mae:.2f} * {worsen_ratio}"
        )
        return False, reasons
    return True, []
