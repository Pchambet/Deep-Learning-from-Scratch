# Lab: Keras baselines on MNIST

The course builds every network in NumPy. The lab answers the obvious follow-up: how do
those ideas fare against a standard framework on a standard benchmark?

| Case study | Model | Test accuracy (MNIST, 10,000 images) | Run |
| --- | --- | --- | --- |
| [`mnist/`](mnist/) | Dense network 784-128-64-10 (Keras) | 96.4% (359 errors), from the notebook's saved output | [notebook](mnist/mnist.ipynb) · `train_mlp.py` |
| [`cnn/`](cnn/) | Two conv + pooling blocks, dense head (Keras) | 98.9% (111 errors), 5 epochs | [notebook](cnn/CNN.ipynb) · `train_cnn.py` |

Both need TensorFlow, which is kept out of the default environment:

```bash
uv sync --group lab
make lab            # runs both training scripts; artefacts go to lab/*/outputs/
```

MNIST is downloaded on first use by `keras.datasets.mnist`.
