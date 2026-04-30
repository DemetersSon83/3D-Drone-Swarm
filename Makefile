.PHONY: install test lint format run-example

install:
	python -m pip install -e ".[dev,viz,parquet]"

test:
	pytest

lint:
	ruff check src tests examples
	mypy src

format:
	ruff format src tests examples
	ruff check --fix src tests examples

run-example:
	python examples/run_basic_swarm.py --steps 100 --n-drones 50 --output outputs/transitions.csv
