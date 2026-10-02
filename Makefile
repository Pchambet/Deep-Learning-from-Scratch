.DEFAULT_GOAL := help
.PHONY: help setup test lint format smoke demo figures check latex lab clean

# Reproducible PDFs: pdfTeX stamps this fixed date (2026-01-01 UTC) instead of
# "now", so rebuilding unchanged sources yields byte-identical files.
export SOURCE_DATE_EPOCH := 1767225600
export FORCE_SOURCE_DATE := 1

# LaTeX guide directory -> published PDF name in pdf/
GUIDES := main:main mnist:mnist cnn:CNN \
	episode_04:All\ Eyes\ on\ You episode_05:The\ Rise\ of\ Intelligence \
	episode_06:Alive episode_07:Horizon\ of\ Depth

help:  ## List the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Create the virtual environment from uv.lock
	uv sync --locked

test:  ## Unit tests, gradient checks and convergence tests
	uv run pytest -q

lint:  ## Ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## Apply ruff fixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

smoke:  ## Train the single neuron for 20 steps on the cat/dog images
	uv run python scripts/smoke_test.py

demo:  ## Episode V: two-layer network on concentric circles (figures in outputs/)
	uv run python scripts/episode_05_demo.py

figures:  ## Regenerate docs/figures/*.png and docs/results.json (~3 min)
	uv run python scripts/make_figures.py

check: lint test smoke demo  ## Everything CI runs

latex:  ## Build every LaTeX guide and copy it to pdf/ (needs latexmk + TeX Live)
	@for entry in $(GUIDES); do \
		dir=$${entry%%:*}; name=$${entry#*:}; tex=$$(ls latex/$$dir/*.tex); \
		echo "Building $$tex -> pdf/$$name.pdf"; \
		(cd latex/$$dir && latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet $$(basename $$tex) >/dev/null) \
			|| { echo "LaTeX failed: see latex/$$dir/*.log"; exit 1; }; \
		cp "$${tex%.tex}.pdf" "pdf/$$name.pdf"; \
	done

lab:  ## Keras baselines on MNIST (installs TensorFlow, downloads MNIST)
	uv sync --group lab
	cd lab/mnist && uv run --group lab python train_mlp.py
	cd lab/cnn && uv run --group lab python train_cnn.py

clean:  ## Remove build artefacts and caches
	cd latex && for dir in */; do (cd $$dir && latexmk -c >/dev/null 2>&1); done; true
	rm -rf outputs lab/*/outputs .pytest_cache .ruff_cache
