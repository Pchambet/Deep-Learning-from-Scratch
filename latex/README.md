# LaTeX sources of the guides

Every PDF in [`pdf/`](../pdf/) except Episodes I–III is built from these sources.

| Folder | Builds | Content |
| --- | --- | --- |
| `main/` | `pdf/main.pdf` | The long guide (117 pages): from one neuron to a network of any depth |
| `mnist/` | `pdf/mnist.pdf` | Dense networks on MNIST |
| `cnn/` | `pdf/CNN.pdf` | Convolutional networks |
| `episode_04/` | `pdf/All Eyes on You.pdf` | Episode IV: training loop on real images |
| `episode_05/` | `pdf/The Rise of Intelligence.pdf` | Episode V: theory of the two-layer network |
| `episode_06/` | `pdf/Alive.pdf` | Episode VI: the two-layer network in code |
| `episode_07/` | `pdf/Horizon of Depth.pdf` | Episode VII: generalising to L layers |

Figures of Episodes VI and VII are produced by `generate_figures.py` in their folder
(`uv run python latex/episode_06/generate_figures.py`); the other guides use the images in
[`assets/photos_git/`](../assets/photos_git/).

## Build

From the repository root:

```bash
make latex
```

This runs `latexmk` on each guide with `-halt-on-error` (a guide that does not compile
cleanly fails the build) and copies the PDFs to `pdf/` under their published names.
`SOURCE_DATE_EPOCH` is fixed, so rebuilding unchanged sources gives byte-identical PDFs.

Requirements: `latexmk` and a TeX Live distribution. A minimal TinyTeX also needs
`tlmgr install tikzfill pdfcol listingsutf8` for `main.tex`.

The sources of Episodes I–III (*Theory of a Neuron*, *The Art of Descent*, *Birth of a
Neuron*) are not in this repository; their PDFs are published as-is.
