# Deep Learning from Scratch — unified build
PYTHON ?= python3

.PHONY: latex latex-main latex-mnist latex-cnn latex-episode05 copy-pdf smoke episode5-demo quality precommit clean

# Build all LaTeX and copy to pdf/
latex: latex-main latex-mnist latex-cnn latex-episode05
	@$(MAKE) copy-pdf

latex-main:
	@echo "Building main.pdf..."
	@cd latex/main && $(MAKE) && cd ../..

latex-mnist:
	@echo "Building mnist.pdf..."
	@cd latex/mnist && $(MAKE) && cd ../..

latex-cnn:
	@echo "Building CNN.pdf..."
	@cd latex/cnn && $(MAKE) && cd ../..

latex-episode05:
	@echo "Building episode_05.pdf..."
	@cd latex/episode_05 && $(MAKE) && cd ../..

copy-pdf:
	@echo "Copying PDFs to pdf/..."
	@test -f latex/main/main.pdf && cp latex/main/main.pdf pdf/ || true
	@test -f latex/mnist/dist/mnist.pdf && cp latex/mnist/dist/mnist.pdf pdf/ || test -f latex/mnist/mnist.pdf && cp latex/mnist/mnist.pdf pdf/ || true
	@test -f latex/cnn/dist/CNN.pdf && cp latex/cnn/dist/CNN.pdf pdf/ || test -f latex/cnn/CNN.pdf && cp latex/cnn/CNN.pdf pdf/ || true
	@test -f latex/episode_04/episode_04.pdf && cp "latex/episode_04/episode_04.pdf" "pdf/All Eyes on You.pdf" || true
	@test -f latex/episode_05/episode_05.pdf && cp "latex/episode_05/episode_05.pdf" "pdf/The Rise of Intelligence.pdf" || true
	@echo "Done."

smoke:
	@echo "Running notebook smoke test..."
	@$(PYTHON) scripts/smoke_test.py

episode5-demo:
	@echo "Running Episode 5 demo..."
	@$(PYTHON) scripts/episode_05_demo.py

quality:
	@echo "Compiling Python sources..."
	@$(PYTHON) -m compileall src lab scripts notebooks/birth_of_a_neuron.py
	@$(MAKE) smoke

precommit:
	@echo "Running pre-commit on all files..."
	@pre-commit run --all-files

clean:
	@cd latex/main && make clean && cd ../..
	@cd latex/mnist && make clean && cd ../..
	@cd latex/cnn && make clean && cd ../..
	@cd latex/episode_05 && make clean && cd ../..
