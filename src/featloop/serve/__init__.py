"""Batch scoring against Production (or Staging) registry model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from featloop import config
from featloop.features import feature_matrix, materialize, time_split
from featloop.registry import load_stage


def batch_score(
    stage: str = "Production",
    out_path: Path | None = None,
    feature_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score holdout/serve window; write predictions parquet."""
    table = feature_table if feature_table is not None else materialize()
    _, _, holdout = time_split(table)
    model = load_stage(stage)
    X, _ = feature_matrix(holdout)
    preds = model.predict(X)
    result = holdout[["entity_id", "event_ts", config.LABEL_COLUMN]].copy()
    result["prediction"] = preds
    result["model_stage"] = stage
    out_path = out_path or config.PREDICTIONS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    holdout.to_parquet(config.SERVE_SNAPSHOT, index=False)
    return result
