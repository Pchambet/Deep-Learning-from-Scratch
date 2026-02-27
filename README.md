<p align="center">
  <img src="assets/banners/banner.png" alt="Deep Learning from Scratch — by Pierre Chambet" width="800">
</p>

<h1 align="center">Deep Learning from Scratch</h1>
<p align="center">
  From first principles to real images — one neuron, one layer, one insight at a time.<br>
  <em>Part of <strong>WIL</strong> — Wide-Range Ideas Laboratory</em><br>
  <a href="https://www.linkedin.com/in/pierre-chambet-289a5b220/">LinkedIn</a> · 
  <a href="https://github.com/Pchambet">GitHub</a>
</p>

---

> “Don’t just run `.fit()`. Build the thing, understand it, and then trust it.”

---

## Deux portes d'entrée

| | **Cours** | **Lab** |
|--|-----------|---------|
| **Pour qui ?** | Apprendre, suivre une progression | Explorer, expérimenter, creuser |
| **Format** | Épisodes PDF + notebooks, linéaire | Case studies, scripts, liberté |
| **Commencer** | [Ep. I](#-linkedin-series--deep-learning-from-scratch-3-épisodes) ou [birth_of_a_neuron](notebooks/birth_of_a_neuron.ipynb) | [lab/mnist](lab/mnist/) ou [lab/cnn](lab/cnn/) |

---

## Cours — Parcours principal

Série pédagogique : **théorie → gradients → code**. Un épisode à la fois.

### Série LinkedIn (3 épisodes)

| Épisode | Titre | Contenu | Lien |
|:-------:|-------|---------|------|
| **I** | *Theory of a Neuron* | PDF 10 p. — fonction linéaire, sigmoid, log-loss | [PDF](pdf/Theory%20of%20a%20Neuron.pdf) |
| **II** | *The Art of Descent* | PDF 12 p. — chain rule, ∂ℓ/∂w, ∂ℓ/∂b | [PDF](pdf/The%20Art%20of%20Descent.pdf) · [Notebook](notebooks/02_gradients_single_neuron.ipynb) |
| **III** | *Birth of a Neuron* | PDF 18 p. + Colab — neurone codé à la main | [PDF](pdf/Birth%20of%20a%20Neuron.pdf) · [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/birth_of_a_neuron.ipynb) |

> Comment **NEURON** (Ep. I), **GRADIENT** (Ep. II) ou **BIRTH** (Ep. III) sur les posts LinkedIn pour recevoir le PDF en DM.  
> #DeepLearningJourney

### Notebooks du cours

| # | Notebook | Focus | Lié à |
|:-:|-----------|--------|-------|
| — | **birth_of_a_neuron** | Neurone codé à la main (toxic plants) | Ep. III · Colab |
| 01 | **Single Neuron** | Linear model, sigmoid | Thème Ep. I |
| 02 | **Gradients Single Neuron** | ∂L/∂w, ∂L/∂b, chain rule | Ep. II |
| 04 | **Training Loop** | Forward → loss → backward → update (cats vs dogs) | — |
| 07 | **Two-Layer Gradients** | Backprop 2 couches | — |
| 08 | **Two-Layer Network** | Réseau 2 couches sur images | — |
| 09 | **Backprop Any Depth** | Backprop L couches | — |
| 11 | **MNIST MLP Baseline** | Dense network, MNIST | — |
| 12 | **MNIST CNN Baseline** | CNN, feature maps | — |

### Guides additionnels (PDF)

| File | Theme |
|------|--------|
| [main.pdf](pdf/main.pdf) | Synthèse globale — neurones à boucle d'apprentissage |
| [mnist.pdf](pdf/mnist.pdf) | Réseaux denses sur MNIST |
| [CNN.pdf](pdf/CNN.pdf) | Comprendre les convolutions |

---

## Lab — Exploration

Pour aller plus loin : case studies, scripts, expérimentations.

👉 **[Voir le Lab](lab/README.md)**

| Projet | Description |
|--------|-------------|
| [**MNIST Case Study**](lab/mnist/) | Pipeline MLP complet, normalisation, training curves |
| [**CNN Case Study**](lab/cnn/) | Convolutions, filtres, pooling sur MNIST |

---

## Quickstart

```bash
git clone https://github.com/Pchambet/Deep-Learning-from-Scratch.git
cd Deep-Learning-from-Scratch
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/birth_of_a_neuron.ipynb
```

**Sans install :** [Colab — birth_of_a_neuron](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/birth_of_a_neuron.ipynb)

---

## Structure du dépôt

```
Deep-Learning-from-Scratch/
├── notebooks/           # Cours — birth_of_a_neuron, 01, 02, 04, 07–09, 11, 12
├── pdf/                 # Guides (Ep. I–III, main, mnist, CNN)
├── lab/                 # Lab — case studies
│   ├── mnist/           # MNIST MLP (notebook + train_mlp.py)
│   └── cnn/             # CNN (notebook + train_cnn.py)
├── src/                 # utilities.py (load_data HDF5)
├── data/                # trainset.hdf5, testset.hdf5 (cats vs dogs)
├── assets/              # figures, banners
├── requirements.txt
└── README.md
```

---

## Philosophie

> "Learning isn't remembering — it's rebuilding."

Pas de boîtes noires. Chaque poids, gradient et mise à jour est tracé.

---

## Pour les recruteurs

**En 5 minutes**, ce dépôt montre que je :
- Comprends les maths derrière les réseaux de neurones
- Implémente et débogue des modèles de bout en bout
- Communique clairement et visuellement
- Apprends en autonomie et livre des résultats propres

**Points d'entrée :**
- [birth_of_a_neuron.ipynb](notebooks/birth_of_a_neuron.ipynb) — clarté
- [02_gradients_single_neuron.ipynb](notebooks/02_gradients_single_neuron.ipynb) — théorie
- [11_mnist_mlp_baseline.ipynb](notebooks/11_mnist_mlp_baseline.ipynb) — application
- [12_mnist_cnn_baseline.ipynb](notebooks/12_mnist_cnn_baseline.ipynb) — maturité

---

## Contribute / Connect

Une erreur, une idée ? Issue ou PR bienvenues.  
Tu apprends en public aussi ? On se connecte.

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
Part of WIL™ — Wide-Range Ideas Laboratory · © 2025 Pierre Chambet
</i></p>
