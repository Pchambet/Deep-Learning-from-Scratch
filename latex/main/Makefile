PDF=main.pdf
TEX=main.tex

all: $(PDF)

$(PDF): $(TEX)
	latexmk -pdf -quiet $(TEX)

clean:
	latexmk -C
