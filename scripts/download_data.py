#!/usr/bin/env python3
"""Optional refresh of UCI Bike Sharing hour.csv. Demo works offline if raw present."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URLS = [
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    data = None
    last_err: Exception | None = None
    for url in URLS:
        try:
            with urlopen(url, timeout=60) as resp:
                data = resp.read()
            break
        except Exception as exc:  # noqa: BLE001 — try next mirror
            last_err = exc
    if data is None:
        raise SystemExit(f"Failed to download dataset: {last_err}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extract("hour.csv", path=RAW)
        if "Readme.txt" in zf.namelist():
            zf.extract("Readme.txt", path=RAW)
    print(f"Wrote {RAW / 'hour.csv'}")


if __name__ == "__main__":
    main()
