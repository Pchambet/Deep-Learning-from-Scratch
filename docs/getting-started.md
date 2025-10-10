# Getting Started

## Run notebooks in Colab

Use the Colab badges inside notebooks, or open them directly from GitHub.

Examples:

- MNIST case study (Notebook): [Open in Colab](https://colab.research.google.com/github/Pchambet/mnist-case-study/blob/main/mnist.ipynb)

## Run locally (macOS Apple Silicon)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Then inside a notebook cell:

```python
import widl
widl.set_seed(42)
```

### Notes

- For TensorFlow on Apple Silicon, prefer `tensorflow-macos` + `tensorflow-metal`.
- If you prefer PyTorch, we can provide Torch-based scripts (MPS support) on request.
