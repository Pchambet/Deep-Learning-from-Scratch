PDF=CNN.pdf
TEX=CNN.tex
OUTDIR=dist
PKG_NAME=guide3_cnn_on_mnist

.PHONY: all clean package

all: $(PDF)

$(PDF): $(TEX)
	@mkdir -p $(OUTDIR)
	@if [ -n "$(WATERMARK)" ]; then \
	  latexmk -pdf -quiet -f -jobname=CNN -pdflatex="pdflatex %O -interaction=nonstopmode '\\def\\WATERMARK{$(WATERMARK)}\\input{$(TEX)}'" $(TEX) || true; \
	else \
	  latexmk -pdf -quiet -f $(TEX) || true; \
	fi; \
	if [ ! -f CNN.pdf ]; then echo 'Build failed: CNN.pdf not generated'; exit 1; fi
	@cp CNN.pdf $(OUTDIR)/

package: $(PDF)
	@mkdir -p $(OUTDIR)
	@zip -r $(OUTDIR)/$(PKG_NAME).zip $(OUTDIR)/CNN.pdf ../../lab/cnn/train_cnn.py ../../lab/cnn/README.md -x "*.DS_Store"

clean:
	latexmk -C
	rm -rf $(OUTDIR)
