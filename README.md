<p align="center">
  <img src="assets/banners/banner.png" alt="Deep Learning from Scratch — by Pierre Chambet" width="800">
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
| **LinkedIn Series (3 épisodes)** | `Theory of a Neuron`, `The Art of Descent`, `Birth of a Neuron` | Cours PDF + Colab — théorie → gradients → code |
| **Notebooks** | Parcours de neurone à CNN (01, 02, 04, 07–09, 11, 12, birth_of_a_neuron) | Implémentations et démos |
| **Guides additionnels** | `main.pdf`, `mnist.pdf`, `CNN.pdf` | Théorie avancée, MNIST, convolutions |

> Les PDFs LinkedIn (3 épisodes) = le *pourquoi* et le *comment* pensée.  
> Les notebooks = le *comment* codé. Les posts partagent le *parcours*.

---

## 🧩 Notebook Index (Chronological Path)

| # | Notebook | Focus | Lié à |
|:-:|-----------|--------|-------|
| — | **birth_of_a_neuron** | Neurone codé à la main (toxic plants) | Ep. III LinkedIn · Colab |
| 01 | **Single Neuron** | Linear model, sigmoid activation | Thème Ep. I |
| 02 | **Gradients Single Neuron** | ∂L/∂w et ∂L/∂b, chain rule | Ep. II LinkedIn |
| 04 | **Training Loop** | Forward → loss → backward → update (cats vs dogs) | — |
| 07 | **Two-Layer Gradients** | Backprop 2 couches, théorie | — |
| 08 | **Two-Layer Network** | Réseau 2 couches sur images | — |
| 09 | **Backprop Any Depth** | Backprop L couches | — |
| 11 | **MNIST MLP Baseline** | Dense network, MNIST | — |
| 12 | **MNIST CNN Baseline** | CNN, feature maps | — |

---

## 📘 Guides (Theory PDFs)

### Série LinkedIn — 3 épisodes (théorie → gradients → code)

| Épisode | File | Theme |
|:-------:|------|--------|
| I | [`Theory of a Neuron.pdf`](pdf/Theory%20of%20a%20Neuron.pdf) | Neurone : fonction linéaire, sigmoid, log-loss |
| II | [`The Art of Descent.pdf`](pdf/The%20Art%20of%20Descent.pdf) | Chain rule, gradients ∂ℓ/∂w, ∂ℓ/∂b |
| III | [`Birth of a Neuron.pdf`](pdf/Birth%20of%20a%20Neuron.pdf) | Neurone codé à la main, toxic plants |

### Guides additionnels

| File | Theme |
|------|--------|
| `main.pdf` | Synthèse globale — neurones à boucle d’apprentissage |
| `mnist.pdf` | Réseaux denses sur MNIST |
| `CNN.pdf` | Comprendre les convolutions |

---

## ⚙️ Quickstart

```bash
git clone https://github.com/Pchambet/deep-learning-from-scratch.git
cd deep-learning-from-scratch
python -m venv .venv && source .venv/bin/activate  # ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt
jupyter lab notebooks/birth_of_a_neuron.ipynb
```

Ou exécuter directement sur [Google Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/birth_of_a_neuron.ipynb) (aucune install).

---

## 🧱 Repository Structure

```
deep-learning-from-scratch/
├── notebooks/           # birth_of_a_neuron, 01, 02, 04, 07–09, 11, 12
├── pdf/                 # Theory of a Neuron, The Art of Descent, Birth of a Neuron, main, mnist, CNN
├── src/                 # utilities.py (load_data pour HDF5)
├── data/                # trainset.hdf5, testset.hdf5 (cats vs dogs)
├── assets/              # figures, banners, photos_git
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📢 LinkedIn Series — Deep Learning From Scratch (3 épisodes)

Série de cours PDF partagés sur [**LinkedIn**](https://www.linkedin.com/in/pierre-chambet-289a5b220/) — théorie → gradients → code.

| Épisode | Titre | Contenu | Lien |
|:-------:|-------|---------|------|
| **I** | *Theory of a Neuron* | PDF 10 p. — linear function, sigmoid, log-loss | [PDF](pdf/Theory%20of%20a%20Neuron.pdf) |
| **II** | *The Art of Descent* | PDF 12 p. — chain rule, ∂ℓ/∂w, ∂ℓ/∂b | [PDF](pdf/The%20Art%20of%20Descent.pdf) · [Notebook](notebooks/02_gradients_single_neuron.ipynb) |
| **III** | *Birth of a Neuron* | PDF 18 p. + Colab — neurone codé à la main | [PDF](pdf/Birth%20of%20a%20Neuron.pdf) · [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/birth_of_a_neuron.ipynb) |

> Comment **NEURON** (Ep. I), **GRADIENT** (Ep. II) ou **BIRTH** (Ep. III) sur les posts LinkedIn pour recevoir le PDF en DM.  
> #DeepLearningJourney

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
- [`birth_of_a_neuron.ipynb`](notebooks/birth_of_a_neuron.ipynb) (clarté — neurone de A à Z)
- [`02_gradients_single_neuron.ipynb`](notebooks/02_gradients_single_neuron.ipynb) (théorie des gradients)
- [`11_mnist_mlp_baseline.ipynb`](notebooks/11_mnist_mlp_baseline.ipynb) (application)
- [`12_mnist_cnn_baseline.ipynb`](notebooks/12_mnist_cnn_baseline.ipynb) (maturité)

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
