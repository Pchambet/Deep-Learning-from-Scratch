# Deep-Learning-from-Scratch

Neural networks rebuilt from first principles in NumPy, from one neuron to any depth:
every gradient derived on paper, coded by hand, and checked against finite differences.

[![CI](https://github.com/Pchambet/Deep-Learning-from-Scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Pchambet/Deep-Learning-from-Scratch/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-0d9488)
[![License: MIT](https://img.shields.io/badge/license-MIT-64748b)](LICENSE)

![Decision boundaries on two interleaved spirals: a single neuron reaches 63.0% held-out accuracy, one hidden layer 89.3%, three hidden layers 99.7%](docs/figures/hero_spirals.png)

## TL;DR

- **The maths is verified, not assumed.** Back-propagation in [`src/deep_network.py`](src/deep_network.py)
  matches central finite differences to a relative error of at most 3.1 × 10⁻⁷ at depths 1 to 4,
  for tanh and sigmoid hidden units. CI re-checks it on every push, along with the agreement of
  the L-layer code with the single neuron of Episode III and the two-layer network of Episode VI.
- **Depth is what bends the boundary.** On two interleaved spirals (300 held-out points), the
  same gradient descent takes accuracy from 63.0% for a single neuron to 89.3% with one hidden
  layer of 16 units and 99.7% with three.
- **Width alone plateaus.** Over five seeds, the best single hidden layer reaches 95.7% (32
  units, 129 parameters) and 64 units drop to 92.5%; two layers of 16 units reach 99.1%
  (337 parameters, worst seed 98.7%).
- **The honest limit.** On 64 × 64 cat/dog photos a fully connected 4096-32-32-1 network gets
  98.5% of its training images right and 57.0% of unseen ones (±6.9 points, 95% interval on 200
  images): it memorises. A small Keras CNN on MNIST makes a third of the errors of a dense
  network (111 vs 359 out of 10,000), which is why convolutions come next.
- **Seven episodes and a 117-page guide**, published as PDFs and rebuilt from their LaTeX
  sources with `make latex`.

## Why it matters

Frameworks make training a network one call to `.fit()`. When that call misbehaves (a loss
that will not move, a model that is perfect on training data and useless after), the person
who has derived and coded back-propagation by hand knows where to look. This repository is
that derivation, written as a course: each step is small enough to check, and each claim
about what a network can or cannot do is backed by a run you can repeat.

## Approach

The course was first published as a LinkedIn series. Each episode pairs a PDF (theory,
derivations, figures) with a notebook (the same ideas in code).

| Episode | PDF | Notebook | What you build |
| :-: | --- | --- | --- |
| I | [Theory of a Neuron](pdf/Theory%20of%20a%20Neuron.pdf) (10 p.) | [`01_single_neuron`](notebooks/01_single_neuron.ipynb) | Linear model, sigmoid, log-loss |
| II | [The Art of Descent](pdf/The%20Art%20of%20Descent.pdf) (12 p.) | [`02_gradients_single_neuron`](notebooks/02_gradients_single_neuron.ipynb) | Chain rule, ∂L/∂w and ∂L/∂b, gradient descent |
| III | [Birth of a Neuron](pdf/Birth%20of%20a%20Neuron.pdf) (18 p.) | [`birth_of_a_neuron`](notebooks/birth_of_a_neuron.ipynb) · [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/birth_of_a_neuron.ipynb) | The neuron coded by hand |
| IV | [All Eyes on You](pdf/All%20Eyes%20on%20You.pdf) (9 p.) | [`04_training_loop_from_scratch`](notebooks/04_training_loop_from_scratch.ipynb) · [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/04_training_loop_from_scratch.ipynb) | Training loop on real images, train/test split |
| V | [The Rise of Intelligence](pdf/The%20Rise%20of%20Intelligence.pdf) (26 p.) | [`05_from_neuron_to_brain`](notebooks/05_from_neuron_to_brain.ipynb) · [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/05_from_neuron_to_brain.ipynb) | Two-layer network: forward and backward pass |
| VI | [Alive](pdf/Alive.pdf) (20 p.) | [`06_alive`](notebooks/06_alive.ipynb) · [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/06_alive.ipynb) | Two-layer network in code; first overfitting |
| VII | [Horizon of Depth](pdf/Horizon%20of%20Depth.pdf) (18 p.) | [`07_horizon_of_depth`](notebooks/07_horizon_of_depth.ipynb) · [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/07_horizon_of_depth.ipynb) | Any number of layers, written with loops |

Going further:

- [**The long guide**](pdf/main.pdf) (117 p.) covers the same path in one document; its deep
  dives are notebooks [`08_two_layer_gradients`](notebooks/08_two_layer_gradients.ipynb)
  (derivation), [`09_two_layer_network`](notebooks/09_two_layer_network.ipynb) and
  [`10_backprop_any_depth`](notebooks/10_backprop_any_depth.ipynb).
- **Guides** on [dense networks for MNIST](pdf/mnist.pdf) and [convolutions](pdf/CNN.pdf), with
  their Keras baselines in [`lab/`](lab/).
- **The tested code**: [`src/deep_network.py`](src/deep_network.py) (L layers),
  [`src/two_layer_network.py`](src/two_layer_network.py) (Episode VI),
  [`notebooks/birth_of_a_neuron.py`](notebooks/birth_of_a_neuron.py) (Episode III) and
  [`src/gradient_check.py`](src/gradient_check.py).

```mermaid
flowchart LR
    A["Derive<br/>PDF episodes"] --> B["Code it in NumPy<br/>notebooks, src/"]
    B --> C["Check it<br/>finite differences, pytest"]
    C --> D["Measure it<br/>held-out data, seeds"]
    D --> E["Find the limit<br/>images need convolutions"]
```

## Results

All numbers below come from `make figures` ([`scripts/make_figures.py`](scripts/make_figures.py)),
which writes [`docs/results.json`](docs/results.json); the run is deterministic.

**Width versus depth.** Five seeds per architecture, same data, learning rate and epochs.
Adding units to a single hidden layer helps up to 32 units, then stops; a second 16-unit layer
is enough to solve the spirals.

![Held-out accuracy against parameter count: one hidden layer peaks at 95.7% with 32 units, two hidden layers of 16 units reach 99.1%](docs/figures/capacity_sweep.png)

| Hidden layer widths | Parameters | Held-out accuracy, mean (min–max over 5 seeds) |
| --- | ---: | --- |
| 16 | 65 | 89.1% (82.7–95.0) |
| 32 | 129 | 95.7% (93.3–97.3) |
| 64 | 257 | 92.5% (86.7–94.7) |
| 16-16 | 337 | 99.1% (98.7–99.3) |
| 16-16-16 | 609 | 99.4% (99.0–99.7) |
| 16-16-16-16 | 881 | 99.5% (99.0–100.0) |

**Where fully connected networks stop.** Trained on the 1,000 cat/dog images of Episodes IV–VII,
the network drives training accuracy to 98.5% while test accuracy hovers between 50% and 60.5%
and ends at 57.0%. A flattened image throws away which pixels are neighbours; the network
can only memorise. Episode VI's notebook shows the same thing with two layers (97.4% train,
52.0% test).

![Train accuracy climbs to 98% while test accuracy stays between 50% and 60.5%](docs/figures/cats_dogs_overfitting.png)

**What convolutions buy** (Keras baselines in [`lab/`](lab/), MNIST test set of 10,000 digits):

| Model | Test accuracy | Errors |
| --- | ---: | ---: |
| Dense 784-128-64-10 ([notebook](lab/mnist/mnist.ipynb)) | 96.4% | 359 |
| Two conv + pooling blocks, dense head ([notebook](lab/cnn/CNN.ipynb)) | 98.9% | 111 |

## Reproduce

```bash
git clone https://github.com/Pchambet/Deep-Learning-from-Scratch.git
cd Deep-Learning-from-Scratch
make setup     # uv sync --locked: Python 3.12 environment from uv.lock
make check     # ruff, 37 tests including gradient checks, smoke test, Episode V demo (~20 s)
make figures   # the experiments above, deterministic (~5 min on a laptop CPU)
make latex     # rebuild every PDF from LaTeX (needs latexmk + TeX Live, ~1 min)
```

Notebooks: `uv run jupyter lab`, or open any Colab link above (no install needed).
Keras baselines: `make lab` (installs TensorFlow, downloads MNIST). The checked-out files take
35 MB; the environment without TensorFlow about 450 MB.

## Repository layout

```text
notebooks/   course notebooks 01-10 + birth_of_a_neuron.py (Episode III functions)
src/         deep_network.py, two_layer_network.py, gradient_check.py, utilities.py
tests/       pytest: shapes, gradient checks, cross-episode agreement, known boundaries
scripts/     make_figures.py (README experiments), smoke_test.py, episode_05_demo.py
docs/        figures/*.png and results.json written by make figures
pdf/         published guides (Episodes I-VII, long guide, MNIST, CNN)
latex/       LaTeX sources of every guide except Episodes I-III
lab/         Keras baselines on MNIST (dense, CNN)
data/        64x64 cat/dog HDF5 files (1,000 train / 200 test), see data/README.md
assets/      images used by the guides and notebooks
```

## Methodology notes and limitations

- **Spiral experiments** use one fixed 700/300 split; seeds change the initial weights only.
  Learning rate (0.5) and epochs (5,000) are the same for every architecture and were not tuned
  per model, so the dip of the 64-unit layer may reflect optimisation rather than capacity.
- **Cat/dog test set**: 200 images, so any accuracy carries about ±7 points of sampling error.
  The figure reports the last epoch; the best test accuracy seen during training (60.5%) is not
  reported as a result because picking it would use the test set for model selection.
- **No validation set, no regularisation, full-batch gradient descent.** These are teaching
  networks: the point is to see each mechanism, not to reach the state of the art.
- The **Keras MNIST dense baseline** (96.4%) is the saved output of its notebook and was not
  re-run for this version; the CNN notebook was executed for it. Notebooks 05 and 06 were
  re-executed; the other notebooks keep the outputs of earlier runs.
- The **cat/dog images** are a small teaching set whose original source and licence are not
  recorded in this repository (see [`data/README.md`](data/README.md)).
- The **LaTeX sources of Episodes I–III** are not in the repository; those three PDFs are
  published as-is.

## References

- I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016, ch. 6
  (feed-forward networks and back-propagation).
- X. Glorot, Y. Bengio, "Understanding the difficulty of training deep feedforward neural
  networks", AISTATS 2010 (the 1/√fan-in initialisation used here).
- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning applied to document
  recognition", Proc. IEEE, 1998 (MNIST, convolutional networks).
- Stanford CS231n course notes, "Gradient checks" (centred differences, relative error).

---

Built by [Pierre Chambet](https://github.com/Pchambet) — decision science for operations under uncertainty.
