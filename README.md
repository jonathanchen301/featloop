# FeatLoop

> FeatLoop turns a public demand dataset into a runnable MLOps loop: materialize features, train and register a model in MLflow, batch-score, then retrain on an event trigger with train/serve skew and accuracy canaries.  
> Clone → `make demo` finishes in under ~10 minutes on a laptop (no cloud account).  
> Built for hiring managers looking for **ML Engineer / ML Platform** candidates who have shipped feature → registry → retrain systems (not notebook demos).

**Not Guardians / AFRL data.** All metrics in this repo are computed on the public [UCI Bike Sharing Dataset](https://doi.org/10.24432/C5W894). Feature names, schemas, and numbers from employer work are intentionally absent.

## Quick start

```bash
git clone <this-repo> && cd featloop
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make demo
```

Warm laptop target: **under ~10 minutes** end-to-end (cold first `pip install` of MLflow dominates). Measured on an Apple Silicon laptop after warm install: **`make demo` ≈ 11–15 seconds** (full cold path including pip still well under 10 minutes).

Inspect registry:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Architecture

```
┌─────────────┐   materialize    ┌──────────────────┐
│ Public CSV  │ ───────────────► │ Offline features │  (Parquet)
│ (bike share │   feature defs   │ entity + time    │
│  extract)   │                  └────────┬─────────┘
└─────────────┘                           │
                                          ▼
                                 Train (sklearn HGBR)
                                          │
                                          ▼
                                 MLflow Registry
                                 None → Staging → Production
                                          │
                                          ▼
                                 Batch score → predictions.parquet
                                          │
                                          ▼
                                 Canaries (PSI skew + high-demand accuracy)

Event path:  events/retrain.trigger  →  make retrain
             (challenger promotes only if canaries pass)
```

| Component | Owns |
|-----------|------|
| `src/featloop/features/` | Definitions, materialize, point-in-time join helpers |
| `src/featloop/train/` | Fit, metrics, MLflow logging |
| `src/featloop/registry/` | Stage transitions, load Production |
| `src/featloop/serve/` | Batch score |
| `src/featloop/monitor/` | PSI + accuracy canary, promote gate |
| `src/featloop/triggers/` | Event file stand-in for upload-triggered retrain |

## Commands

| Target | What it does |
|--------|----------------|
| `make demo` | Materialize → train → Production → score → retrain → canary |
| `make train` | Train + register Staging |
| `make score` | Batch-score Production holdout |
| `make retrain` | Event trigger + challenger + promote-only-if-pass |
| `make canary` | Skew + accuracy against Production |
| `make test` | Unit tests (features, PSI, retrain gate) |

## Label + metrics (frozen for v1)

- **Task:** regression on hourly ridership `cnt` (MAE / RMSE on a **time-based** valid split).
- **Canary:** high-demand binary accuracy (label ≥ train 75th percentile of `cnt`) plus train-vs-serve **PSI** on feature columns.
- **Promote gate:** fail-closed if max PSI (weather/lag features vs recent-train window) ≥ `5.0` (severe / broken-recipe), high-demand accuracy < `0.55`, or challenger MAE > Production MAE × `1.15`. Mild seasonal drift (often visible on `temp`/`atemp` in the canary plot) is expected on a late holdout and does not block by itself.

## Failure modes

| Failure | What happens | Mitigation |
|---------|--------------|------------|
| Train/serve skew | Score-time features drift from train | PSI canary; fail closed on promote |
| Time leakage | Random split leaks future into train | Time-based split only |
| Delayed labels | Real systems get labels late | Demo uses contemporaneous public labels; not a claim about production label SLAs |
| Weak canary | Always-pass gate | Tests force bad distributions / bad accuracy and expect reject |
| MLflow confusion | Miss the UI | `make demo` prints `mlflow ui --backend-store-uri ./mlruns` |
| Honesty breach | Resume metrics pasted as project results | README + LICENSE-DATA: project metrics ≠ Guardians numbers |

## Trust boundaries (local demo)

- **`mlruns/` is a trust boundary.** `mlflow.sklearn.load_model` uses pickle under the hood. Only load models you trained in this repo; do not point the tracking URI at an untrusted or shared volume. Prefer `mlflow ui` bound to localhost.
- **Vendored `data/raw/` is the happy path.** `scripts/download_data.py` is optional refresh only: HTTPS allowlist (`archive.ics.uci.edu`), no redirect following, 50 MiB size cap, and extract paths constrained under `data/raw/`.
- **No secrets in-repo.** `.env` is gitignored; do not commit credentials.

## Data

Vendored under `data/raw/hour.csv` with attribution in [`data/raw/LICENSE-DATA.md`](data/raw/LICENSE-DATA.md). Optional refresh: `python scripts/download_data.py` (demo does **not** require network when raw is present).

## License

MIT for code. Dataset remains under UCI / original author terms — see `LICENSE-DATA.md`.
