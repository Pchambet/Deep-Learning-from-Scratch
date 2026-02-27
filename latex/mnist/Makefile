PDF=mnist.pdf
TEX=mnist.tex
OUTDIR=dist
PKG_NAME=guide2_mlp_on_mnist

.PHONY: all clean package

all: $(PDF)

$(PDF): $(TEX)
	@mkdir -p $(OUTDIR)
	@if [ -n "$(WATERMARK)" ]; then \
	  latexmk -pdf -quiet -f -jobname=mnist -pdflatex="pdflatex %O -interaction=nonstopmode '\\def\\WATERMARK{$(WATERMARK)}\\input{$(TEX)}'" $(TEX) || true; \
	else \
	  latexmk -pdf -quiet -f $(TEX) || true; \
	fi; \
	if [ ! -f mnist.pdf ]; then echo 'Build failed: mnist.pdf not generated'; exit 1; fi
	@cp mnist.pdf $(OUTDIR)/

package: $(PDF)
	@mkdir -p $(OUTDIR)
	@zip -r $(OUTDIR)/$(PKG_NAME).zip $(OUTDIR)/mnist.pdf ../../lab/mnist/train_mlp.py ../../lab/mnist/README.md -x "*.DS_Store"

clean:
	latexmk -C
	rm -rf $(OUTDIR)
