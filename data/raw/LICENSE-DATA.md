# Data license — UCI Bike Sharing Dataset

This repository vendors an extract of the **Bike Sharing Dataset** from the
UCI Machine Learning Repository for offline demo use.

## Source

- Fanaee-T, Hadi. (2013). Bike Sharing Dataset.
  UCI Machine Learning Repository. https://doi.org/10.24432/C5W894
- Original paper: Fanaee-T, H., & Gama, J. (2013). Event labeling combining
  ensemble detectors and background knowledge. Progress in Artificial
  Intelligence.

## Attribution

Cite UCI and the authors above when redistributing or referencing this data.
The raw files under `data/raw/` are **not** Guardians, AFRL, or any employer
proprietary data. FeatLoop metrics are computed only on this public extract.

## Refresh

Optional: `python scripts/download_data.py` re-fetches from UCI if you need a
fresh copy. The demo works offline when `data/raw/hour.csv` is present.
