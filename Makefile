# Deep Learning from Scratch — unified build
.PHONY: latex latex-main latex-mnist latex-cnn copy-pdf clean

# Build all LaTeX and copy to pdf/
latex: latex-main latex-mnist latex-cnn
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

copy-pdf:
	@echo "Copying PDFs to pdf/..."
	@test -f latex/main/main.pdf && cp latex/main/main.pdf pdf/ || true
	@test -f latex/mnist/dist/mnist.pdf && cp latex/mnist/dist/mnist.pdf pdf/ || test -f latex/mnist/mnist.pdf && cp latex/mnist/mnist.pdf pdf/ || true
	@test -f latex/cnn/dist/CNN.pdf && cp latex/cnn/dist/CNN.pdf pdf/ || test -f latex/cnn/CNN.pdf && cp latex/cnn/CNN.pdf pdf/ || true
	@echo "Done."

clean:
	@cd latex/main && make clean && cd ../..
	@cd latex/mnist && make clean && cd ../..
	@cd latex/cnn && make clean && cd ../..
