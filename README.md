<p align="center">
  <img src="banner.png" alt="Deep Learning from Scratch — by Pierre Chambet" width="800">
</p>

<h1 align="center">Deep Learning from Scratch</h1>
<p align="center">
  From first principles to real images — one neuron, one layer, one insight at a time.<br>
  <a href="https://www.linkedin.com/in/pierre-chambet-289a5b220/">LinkedIn</a> • 
  <a href="https://github.com/Pchambet">GitHub</a>
</p>

---

> “Don’t just run `.fit()`. Build the thing, understand it, and then trust it.”

---

## 👋 About This Project

I’m **Pierre Chambet**, a data & deep learning student-engineer who decided to rebuild Deep Learning from scratch —  
not by copying frameworks, but by *understanding every equation, line, and gradient*.

This repository is my **learning-in-public laboratory**.  
It documents the full path from a **hand-coded neuron** in NumPy to a **convolutional network** on MNIST —  
all explained, derived, and visualized with care.

It’s both a **portfolio of understanding** and a **teaching resource**:  
math → code → intuition → result.

---

## 🧭 Project Architecture

| Layer | Content | Purpose |
|--------|----------|----------|
| **PDF Guides** | `main_capstone.pdf`, `mnist_guide.pdf`, `cnn_guide.pdf` | Theoretical backbone and narrative |
| **Notebooks (01–12)** | Full implementations, from neuron → CNN | Code + visual demonstrations |
| **LinkedIn Series** | Weekly public lessons | Outreach, credibility, reflection |

> The PDFs tell the *why*, the notebooks show the *how*,  
> and the posts share the *journey*.

---

## 🧩 Notebook Index (Chronological Path)

| # | Notebook | Focus | Output |
|:-:|-----------|--------|---------|
| 01 | **Single Neuron** | Linear model, sigmoid activation | Decision boundary |
| 02 | **Log-Loss & Metrics** | Binary cross-entropy, clipping, accuracy | Loss curve |
| 03 | **Gradients by Hand** | ∂L/∂w and ∂L/∂b derivation | Gradient sanity check |
| 04 | **Training Loop** | Forward → loss → backward → update | Accuracy over time |
| 05 | **Image Pipeline** | Load & normalize data (HDF5 or MNIST) | Sample grid |
| 06 | **From Scratch on Images** | Apply hand-built loop to real pixels | Training curve |
| 07 | **Two-Layer Gradients** | Derive and visualize 2-layer backprop | Equations & schema |
| 08 | **Two-Layer Network** | Implement full 2-layer NN | Non-linear boundary |
| 09 | **Backprop Any Depth** | General L-layer backprop (looped) | Gradient flow |
| 10 | **Decision Boundaries** | Moons / Circles / Blobs | Boundary comparison |
| 11 | **MNIST MLP Baseline** | Dense network + error analysis | Confusion matrix |
| 12 | **MNIST CNN Baseline** | Convolutional net + feature maps | Learned filters |

---

## 📘 Guides (Theory PDFs)

| File | Theme | Role |
|------|--------|------|
| `main_capstone.pdf` | **Fundamentals & Training Logic** | The full story — neurons, gradients, learning loop |
| `mnist_guide.pdf` | **Dense Networks on MNIST** | How to move from vectors to real handwritten digits |
| `cnn_guide.pdf` | **Understanding Convolutions** | Why spatial structure changes everything |

> These PDFs are not static papers — they mirror the notebooks and serve as theoretical anchors.

---

## ⚙️ Quickstart

```bash
git clone https://github.com/Pchambet/deep-learning-from-scratch.git
cd deep-learning-from-scratch
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
jupyter lab notebooks/01_single_neuron.ipynb
```

---

## 🧱 Repository Structure

```
deep-learning-from-scratch/
├── notebooks/           # 01–12 notebooks (chronological learning path)
├── pdf/                 # main_capstone.pdf, mnist_guide.pdf, cnn_guide.pdf
├── src/                 # helper code (e.g., utilities.py)
├── assets/
│   ├── figures/         # exported plots (decision boundaries, confusion matrices)
│   └── banners/         # repo and LinkedIn visuals
├── env/                 # requirements and environment files
├── README.md
└── LICENSE
```

---

## 📢 LinkedIn Series — #DeepLearningJourney

Every notebook becomes a short, visual lesson shared on [**LinkedIn**](https://www.linkedin.com/in/pierre-chambet-289a5b220/).  
Each post includes 1 idea, 1 plot, and 1 link to the corresponding notebook.

| Episode | Title | Notebook |
|:--------|:-------|:----------|
| 1 | *I built a neuron from scratch* | [`01_single_neuron.ipynb`](notebooks/01_single_neuron.ipynb) |
| 2 | *Log-loss explained in 60 seconds* | [`02_logloss_and_metrics.ipynb`](notebooks/02_logloss_and_metrics.ipynb) |
| 3 | *How backprop really works* | [`03_gradients_single_neuron.ipynb`](notebooks/03_gradients_single_neuron.ipynb) |
| 4 | *A training loop that actually learns* | [`04_training_loop_from_scratch.ipynb`](notebooks/04_training_loop_from_scratch.ipynb) |
| 5 | *From vectors to images — MNIST* | [`11_mnist_mlp_baseline.ipynb`](notebooks/11_mnist_mlp_baseline.ipynb) |
| 6 | *When the network starts to see — CNNs* | [`12_mnist_cnn_baseline.ipynb`](notebooks/12_mnist_cnn_baseline.ipynb) |
| 7 | *The big picture: from neuron to CNN* | [`pdf/main_capstone.pdf`](pdf/main_capstone.pdf) |

---

## 🧠 Philosophy

> “Learning isn’t remembering — it’s rebuilding.”

No shortcuts, no black boxes.  
Every weight, bias, and gradient is traced.  
This is **real deep learning** — in both name and process.

---

## 🧾 For Recruiters

**In five minutes**, this repo tells you that I:
- Understand the math behind neural networks.  
- Can implement and debug deep learning models end-to-end.  
- Communicate complex ideas clearly and visually.  
- Learn independently, structure work, and deliver clean results.

Start with:
- `01_single_neuron.ipynb` (clarity)
- `04_training_loop_from_scratch.ipynb` (method)
- `11_mnist_mlp_baseline.ipynb` (application)
- `12_mnist_cnn_baseline.ipynb` (maturity)

---

## 🤝 Contribute / Connect

If you find an error or idea worth exploring, open an issue or PR.  
If you’re learning in public too, tag me — let’s connect.

<p align="center">
  <a href="https://www.linkedin.com/in/pierre-chambet-289a5b220/">
    <img src="https://img.shields.io/badge/Follow%20on%20LinkedIn-blue?style=flat-square&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="https://github.com/Pchambet">
    <img src="https://img.shields.io/badge/Explore%20more%20projects-black?style=flat-square&logo=github" alt="GitHub">
  </a>
</p>

---

<p align="center"><i>
Deep Learning from Scratch — built with patience, mathematics, and curiosity.<br>
© 2025 Pierre Chambet. All rights reserved.
</i></p>
