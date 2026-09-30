"""Event-file trigger stand-in for upload-triggered retrain."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from featloop import config


def ensure_trigger(path: Path | None = None, reason: str = "demo_event") -> Path:
    path = path or config.RETRAIN_TRIGGER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"reason: {reason}\ncreated_at: {datetime.now(timezone.utc).isoformat()}\n"
    )
    return path


def trigger_present(path: Path | None = None) -> bool:
    path = path or config.RETRAIN_TRIGGER
    return path.exists()


def consume_trigger(path: Path | None = None) -> str | None:
    """Read and remove trigger file; return contents if present."""
    path = path or config.RETRAIN_TRIGGER
    if not path.exists():
        return None
    text = path.read_text()
    path.unlink()
    return text
