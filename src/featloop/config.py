"""Paths and thresholds for FeatLoop (local-only, no cloud)."""

from __future__ import annotations

from pathlib import Path

# Repo root: src/featloop/config.py → parents[2]
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FEATURE_DEFS = ROOT / "features" / "definitions.yml"
FEATURE_TABLE = DATA_PROCESSED / "features.parquet"
TRAIN_SNAPSHOT = DATA_PROCESSED / "train_features.parquet"
SERVE_SNAPSHOT = DATA_PROCESSED / "serve_features.parquet"
PREDICTIONS = DATA_PROCESSED / "predictions.parquet"
REPORTS = ROOT / "reports"
EVENTS = ROOT / "events"
RETRAIN_TRIGGER = EVENTS / "retrain.trigger"
MLRUNS = ROOT / "mlruns"

EXPERIMENT_NAME = "featloop-demand"
REGISTERED_MODEL_NAME = "featloop-demand-regressor"

# Time-based split: train on earlier hours, hold out last window for canary
TRAIN_FRAC = 0.80
VALID_FRAC = 0.10  # of remaining after train → valid; rest = holdout/serve

FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "cnt_lag_1h",
    "cnt_roll_24h_mean",
    "is_rush_hour",
]

# Skew canary ignores pure calendar keys (train vs late holdout always shifts on yr/mnth).
# Gate focuses on weather / lag features where train/serve recipe bugs show up.
PSI_FEATURE_COLUMNS = [
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "cnt_lag_1h",
    "cnt_roll_24h_mean",
    "is_rush_hour",
    "holiday",
    "workingday",
]

LABEL_COLUMN = "cnt"
HIGH_DEMAND_QUANTILE = 0.75  # for binary canary accuracy

# Canary gates (documented; tuned for public extract stability)
PSI_WARN = 0.25
PSI_FAIL = 5.0  # severe skew / broken recipe; seasonal temp drift on holdout is ~3–4
ACCURACY_MIN = 0.55  # high-demand classification on holdout; fail-closed below
MAE_WORSEN_RATIO = 1.15  # challenger MAE must not exceed prod * ratio

RANDOM_SEED = 42
