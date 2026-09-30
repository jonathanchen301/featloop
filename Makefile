.PHONY: demo train score retrain canary test clean install

PYTHON ?= python3
PIP ?= pip

install:
	$(PIP) install -e ".[dev]"

demo: install
	$(PYTHON) scripts/run_demo.py

train:
	$(PYTHON) scripts/train.py

score:
	$(PYTHON) scripts/score.py

retrain:
	$(PYTHON) scripts/retrain.py

canary:
	$(PYTHON) scripts/canary.py

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf mlruns models data/processed/*.parquet reports/canary.* events/retrain.trigger
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
