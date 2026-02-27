# LaTeX sources

Source files for the PDF guides. Build locally to regenerate PDFs and keep a consistent visual style across all materials.

## Structure

| Folder | Source | Output | Description |
|--------|--------|--------|-------------|
| `main/` | main.tex | main.pdf | Full fundamentals guide — neurons to training loop |
| `mnist/` | mnist.tex | mnist.pdf | Dense networks on MNIST |
| `cnn/` | CNN.tex | CNN.pdf | Convolutional neural networks |

## Build

From each subfolder:

```bash
cd latex/main   # or mnist, or cnn
make
```

Requires `latexmk` and a LaTeX distribution (e.g. MacTeX, TeX Live).

## Copy to pdf/

After building, copy the PDFs to the root `pdf/` folder:

```bash
cp latex/main/main.pdf pdf/
cp latex/mnist/dist/mnist.pdf pdf/
cp latex/cnn/dist/CNN.pdf pdf/
```

Or use the top-level Makefile (if present) to automate this.
