# MNIST with a dense network (Keras)

A multilayer perceptron on handwritten digits: normalisation, training curves, confusion
matrix and the misclassified digits. Keras baseline for comparison with the from-scratch
networks of the course.

```bash
uv sync --group lab
uv run --group lab jupyter lab mnist.ipynb      # notebook
cd lab/mnist && uv run --group lab python train_mlp.py   # script, outputs in outputs/
```

[Open in Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/lab/mnist/mnist.ipynb)
