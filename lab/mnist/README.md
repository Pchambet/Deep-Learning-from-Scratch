# MNIST Case Study

An MLP on the MNIST dataset — from normalization to training, step by step.

## Run it

**Notebook:**

```bash
jupyter notebook mnist.ipynb
```

Or [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/lab/mnist/mnist.ipynb)

**Script:**

```bash
make mlp
```

Outputs are saved to `outputs/`.

## Dependencies

Use the root `requirements.txt`, or:

```bash
pip install tensorflow numpy matplotlib
```

On Apple Silicon: `tensorflow-macos` + `tensorflow-metal`.

## Dataset

MNIST is downloaded automatically via `tensorflow.keras.datasets.mnist`.
