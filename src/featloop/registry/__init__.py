"""MLflow Model Registry helpers (local file store)."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from featloop import config


def _client() -> MlflowClient:
    config.MLRUNS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLRUNS.as_uri())
    return MlflowClient(tracking_uri=config.MLRUNS.as_uri())


def list_versions(name: str | None = None) -> list[dict[str, Any]]:
    client = _client()
    name = name or config.REGISTERED_MODEL_NAME
    versions = client.search_model_versions(f"name='{name}'")
    return [
        {
            "version": int(v.version),
            "stage": v.current_stage,
            "run_id": v.run_id,
            "status": v.status,
        }
        for v in sorted(versions, key=lambda x: int(x.version))
    ]


def latest_version(name: str | None = None) -> int | None:
    versions = list_versions(name)
    if not versions:
        return None
    return max(v["version"] for v in versions)


def transition_latest(stage: str, name: str | None = None) -> int:
    """Move the highest version into `stage` (Archive prior occupants of that stage)."""
    client = _client()
    name = name or config.REGISTERED_MODEL_NAME
    version = latest_version(name)
    if version is None:
        raise RuntimeError(f"No registered versions for {name}")
    # Archive existing models in target stage
    for v in list_versions(name):
        if v["stage"] == stage and v["version"] != version:
            client.transition_model_version_stage(
                name=name,
                version=str(v["version"]),
                stage="Archived",
                archive_existing_versions=False,
            )
    client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage=stage,
        archive_existing_versions=False,
    )
    return version


def transition_version(version: int, stage: str, name: str | None = None) -> None:
    client = _client()
    name = name or config.REGISTERED_MODEL_NAME
    if stage == "Production":
        for v in list_versions(name):
            if v["stage"] == "Production" and v["version"] != version:
                client.transition_model_version_stage(
                    name=name,
                    version=str(v["version"]),
                    stage="Archived",
                    archive_existing_versions=False,
                )
    client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage=stage,
        archive_existing_versions=False,
    )


def load_stage(stage: str = "Production", name: str | None = None):
    name = name or config.REGISTERED_MODEL_NAME
    uri = f"models:/{name}/{stage}"
    config.MLRUNS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLRUNS.as_uri())
    return mlflow.sklearn.load_model(uri)


def promote_to_production(version: int, name: str | None = None) -> None:
    transition_version(version, "Production", name=name)
