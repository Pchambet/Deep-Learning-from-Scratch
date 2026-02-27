# Episode 4 — Cat vs Dog (standalone)
PDF=episode_04.pdf
TEX=episode_04.tex

all: $(PDF)

$(PDF): $(TEX)
	latexmk -pdf -quiet $(TEX)

clean:
	latexmk -C
