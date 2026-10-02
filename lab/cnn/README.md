# MNIST with a convolutional network (Keras)

Two convolution + max-pooling blocks and a dense head: 98.9% test accuracy after five
epochs (111 errors on 10,000 digits), a third of the errors of the dense network in
[`../mnist`](../mnist/).

```bash
uv sync --group lab
uv run --group lab jupyter lab CNN.ipynb       # notebook
cd lab/cnn && uv run --group lab python train_cnn.py    # script, outputs in outputs/
```

The companion guide is [`pdf/CNN.pdf`](../../pdf/CNN.pdf).
