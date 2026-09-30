"""Train sklearn demand model and log to local MLflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from featloop import config
from featloop.features import feature_matrix, materialize, time_split


@dataclass
class TrainResult:
    run_id: str
    mae: float
    rmse: float
    n_train: int
    n_valid: int
    model_uri: str


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def train_model(
    *,
    register: bool = True,
    stage_after: str | None = "Staging",
    run_name: str = "train",
) -> TrainResult:
    """Materialize features, fit HGBR, log params/metrics/model to MLflow."""
    config.MLRUNS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLRUNS.as_uri())
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    table = materialize()
    train, valid, holdout = time_split(table)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_parquet(config.TRAIN_SNAPSHOT, index=False)
    holdout.to_parquet(config.SERVE_SNAPSHOT, index=False)

    X_train, y_train = feature_matrix(train)
    X_valid, y_valid = feature_matrix(valid)

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=120,
        random_state=config.RANDOM_SEED,
    )

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        mae, rmse = _metrics(y_valid, pred)

        mlflow.log_params(
            {
                "model": "HistGradientBoostingRegressor",
                "max_depth": 6,
                "learning_rate": 0.08,
                "max_iter": 120,
                "train_frac": config.TRAIN_FRAC,
                "n_features": len(config.FEATURE_COLUMNS),
                "split": "time",
            }
        )
        mlflow.log_metrics({"valid_mae": mae, "valid_rmse": rmse})
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.REGISTERED_MODEL_NAME if register else None,
        )
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

    if register and stage_after:
        from featloop.registry import transition_latest

        transition_latest(stage_after)

    return TrainResult(
        run_id=run_id,
        mae=mae,
        rmse=rmse,
        n_train=len(train),
        n_valid=len(valid),
        model_uri=model_uri,
    )


def evaluate_mae(model: Any, df) -> float:
    X, y = feature_matrix(df)
    pred = model.predict(X)
    return float(mean_absolute_error(y, pred))
