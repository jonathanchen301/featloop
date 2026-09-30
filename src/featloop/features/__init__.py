"""Offline feature materialization + point-in-time helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from featloop import config


def load_definitions(path: Path | None = None) -> dict:
    path = path or config.FEATURE_DEFS
    with path.open() as f:
        return yaml.safe_load(f)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or (config.DATA_RAW / "hour.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/download_data.py or vendor hour.csv."
        )
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["event_ts"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("event_ts").reset_index(drop=True)
    return df


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time safe lags: shift before rolling so labels do not leak."""
    out = df.copy()
    out["cnt_lag_1h"] = out["cnt"].shift(1)
    out["cnt_roll_24h_mean"] = out["cnt"].shift(1).rolling(24, min_periods=1).mean()
    out["is_rush_hour"] = out["hr"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    out["entity_id"] = (
        out["dteday"].dt.strftime("%Y-%m-%d") + "_" + out["hr"].astype(str).str.zfill(2)
    )
    return out


def materialize(raw: pd.DataFrame | None = None, out_path: Path | None = None) -> pd.DataFrame:
    """Build offline feature table and write Parquet."""
    raw = load_raw() if raw is None else raw
    feats = _lag_features(raw)
    # Drop first row with null lag; keep rest
    feats = feats.dropna(subset=["cnt_lag_1h"]).reset_index(drop=True)
    cols = (
        ["entity_id", "event_ts", "dteday", "hr"]
        + config.FEATURE_COLUMNS
        + [config.LABEL_COLUMN]
    )
    # Deduplicate feature list while preserving order (hr appears twice otherwise)
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    table = feats[ordered].copy()
    out_path = out_path or config.FEATURE_TABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    return table


def point_in_time_join(
    entities: pd.DataFrame,
    feature_table: pd.DataFrame,
    entity_keys: list[str] | None = None,
    ts_col: str = "event_ts",
) -> pd.DataFrame:
    """
    As-of join: for each entity row, take the latest feature row with
    feature.event_ts <= entity.event_ts matching keys.
    """
    entity_keys = entity_keys or ["entity_id"]
    left = entities.sort_values(ts_col)
    right = feature_table.sort_values(ts_col)
    # pandas merge_asof requires single key; for entity_id we merge on that + asof ts
    if entity_keys == ["entity_id"]:
        merged = pd.merge_asof(
            left,
            right,
            on=ts_col,
            by="entity_id",
            direction="backward",
            suffixes=("", "_feat"),
        )
        return merged
    # Fallback: keyless asof (demo / tests)
    return pd.merge_asof(left, right, on=ts_col, direction="backward", suffixes=("", "_feat"))


def time_split(
    table: pd.DataFrame,
    train_frac: float = config.TRAIN_FRAC,
    valid_frac: float = config.VALID_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / valid / holdout (serve) split — no random shuffle."""
    n = len(table)
    t_end = int(n * train_frac)
    v_end = t_end + int(n * valid_frac)
    train = table.iloc[:t_end].copy()
    valid = table.iloc[t_end:v_end].copy()
    holdout = table.iloc[v_end:].copy()
    return train, valid, holdout


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[config.FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[config.LABEL_COLUMN].to_numpy(dtype=float)
    return X, y
