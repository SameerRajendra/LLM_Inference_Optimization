PYTHON := python3
VENV   := .venv
PIP    := $(VENV)/bin/pip

.PHONY: venv install install-jax install-vllm install-all build test bench bench-64k bench-128k profile-nsys profile-ncu clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: venv
	MAX_JOBS=8 $(PIP) install -r requirements.txt
	$(PIP) install -e . --no-build-isolation

install-jax:
	$(PIP) install -r requirements-jax.txt

install-vllm:
	$(PIP) install -r requirements-vllm.txt

install-all: install install-jax install-vllm
	$(PIP) install -e ".[dev]" --no-build-isolation

build:
	$(PIP) install -e . --no-build-isolation

test:
	$(VENV)/bin/pytest tests/ -v

bench-64k:
	$(VENV)/bin/python benchmarks/run_benchmarks.py --ctx-len 65536 --top-k 512

bench-128k:
	$(VENV)/bin/python benchmarks/run_benchmarks.py --ctx-len 131072 --top-k 512

profile-nsys:
	nsys profile --trace=cuda,nvtx --output=profiles/nsys_report \
	    $(VENV)/bin/python benchmarks/run_benchmarks.py

profile-ncu:
	ncu --target-processes all --set full --output profiles/ncu_report \
	    $(VENV)/bin/python benchmarks/run_benchmarks.py

clean:
	rm -rf build/ dist/ *.egg-info **/__pycache__ .pytest_cache
	find . -name "*.so" -delete