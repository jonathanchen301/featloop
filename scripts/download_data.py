#!/usr/bin/env python3
"""Optional refresh of UCI Bike Sharing hour.csv. Demo works offline if raw present."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ALLOWED_HOSTS = frozenset({"archive.ics.uci.edu"})
URLS = [
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
]
MAX_BYTES = 50 * 1024 * 1024  # 50 MiB cap


class _NoRedirect(Exception):
    """Raised when a redirect would leave the allowlisted host path uncontrolled."""


def _fetch(url: str) -> bytes:
    """Fetch URL without following redirects; host must stay on the allowlist."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Refusing non-allowlisted URL: {url}")

    opener = build_opener()
    # Drop default redirect handler so Location cannot jump hosts silently.
    opener.handlers = [
        h for h in opener.handlers if h.__class__.__name__ != "HTTPRedirectHandler"
    ]
    req = Request(url, method="GET", headers={"User-Agent": "featloop-download/0.1"})
    try:
        with opener.open(req, timeout=60) as resp:
            final = urlparse(resp.geturl())
            if final.hostname not in ALLOWED_HOSTS:
                raise _NoRedirect(f"Redirected off allowlist to {resp.geturl()}")
            chunks: list[bytes] = []
            total = 0
            while True:
                block = resp.read(1024 * 256)
                if not block:
                    break
                total += len(block)
                if total > MAX_BYTES:
                    raise RuntimeError(f"Download exceeded {MAX_BYTES} bytes")
                chunks.append(block)
            return b"".join(chunks)
    except HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            raise _NoRedirect(f"Redirect not followed ({exc.code}) for {url}") from exc
        raise


def _safe_extract(zf: zipfile.ZipFile, member: str, dest: Path) -> Path:
    dest = dest.resolve()
    # ZipInfo.filename must be a simple basename under dest (no directories / ..).
    name = Path(member).name
    if name != member and "/" in member.replace("\\", "/"):
        raise RuntimeError(f"Refusing nested zip member: {member}")
    out = (dest / name).resolve()
    if not out.is_relative_to(dest):
        raise RuntimeError(f"Refusing zip member outside {dest}: {member}")
    # Extract via ZipInfo with sanitized filename
    info = zf.getinfo(member)
    target = dest / name
    with zf.open(info) as src, target.open("wb") as dst:
        dst.write(src.read())
    return target


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    data = None
    last_err: Exception | None = None
    for url in URLS:
        try:
            data = _fetch(url)
            break
        except (URLError, HTTPError, OSError, ValueError, _NoRedirect, RuntimeError) as exc:
            last_err = exc
    if data is None:
        raise SystemExit(f"Failed to download dataset: {last_err}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        hour = _safe_extract(zf, "hour.csv", RAW)
        if "Readme.txt" in zf.namelist():
            _safe_extract(zf, "Readme.txt", RAW)
    print(f"Wrote {hour}")


if __name__ == "__main__":
    main()
