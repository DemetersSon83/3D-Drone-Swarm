PYTHON ?= python
PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))
OUTPUT_ROOT ?= outputs/swarm_dataset_v1

.PHONY: install install-core test test-core lint format run-example run-smoke run-parquet-smoke run-pilot catalog validate-dataset clean

install-core:
	$(PYTHON) -m pip install -e .

install:
	$(PYTHON) -m pip install -e ".[dev,pipeline,viz]"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

test-core:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -m "not pipeline"

lint:
	$(PYTHON) -m ruff check src tests examples scripts
	$(PYTHON) -m ruff format --check src tests examples scripts
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m ruff format src tests examples scripts
	$(PYTHON) -m ruff check --fix src tests examples scripts

run-example:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) examples/run_basic_swarm.py --steps 100 --n-drones 50 --output outputs/transitions.csv

run-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m drone_swarm.cli run --config configs/smoke/smoke_split.yaml --seed 7 --output-root outputs/smoke --force

run-parquet-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m drone_swarm.cli run --config configs/smoke/smoke_split.yaml --seed 0 --output-root outputs/smoke-parquet --formats parquet --force --validation-level full

run-pilot:
	PYTHON_BIN=$(PYTHON) OUTPUT_ROOT=$(OUTPUT_ROOT) scripts/run_phase1_pilot.sh

catalog:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m drone_swarm.cli catalog --output-root $(OUTPUT_ROOT)

validate-dataset:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_dataset.py --output-root $(OUTPUT_ROOT) --validation-level standard

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
