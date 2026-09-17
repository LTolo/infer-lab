.PHONY: help install verify run test bench lint kernels clean

help:
	@echo "infer-lab"
	@echo "  make install   install the package with dev extras"
	@echo "  make verify    verify the whole project runs error-free"
	@echo "  make run       start the app and all its dependencies"
	@echo "  make test      run the test suite only"
	@echo "  make bench     benchmark with regression tracking"
	@echo "  make kernels   build native extensions and compare backends"
	@echo "  make lint      ruff check"
	@echo "  make clean     remove build artifacts"

install:
	pip install -e ".[dev]"

verify:
	python scripts/verify.py

run:
	python scripts/run_stack.py

test:
	python -m pytest tests/ -q

bench:
	python -m infer_lab.cli bench --track --json

kernels:
	python -m infer_lab.kernels.build -v
	python -m infer_lab.cli kernels

lint:
	ruff check src tests scripts

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache artifacts
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.so" -o -name "*.pyd" -o -name "*.dll" | xargs rm -f 2>/dev/null || true
