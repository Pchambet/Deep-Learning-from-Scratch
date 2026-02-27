# Audit : LinkedIn ↔ GitHub — Alignement et cohérence

**Date :** 27 février 2026  
**Objectif :** Remettre en lien les 3 épisodes LinkedIn (cours PDF) avec le dépôt GitHub, clarifier les correspondances et identifier les écarts.

**Mise à jour (merge WIL) :** Fusion avec le Lab WIL — structure Cours + Lab, case studies MNIST/CNN intégrés, imports corrigés (birth_of_a_neuron, src.utilities).

---

## 1. Ta série LinkedIn réelle — Les 3 épisodes

Tu as construit une **série pédagogique en 3 temps** avec une progression claire :

| Épisode | Titre | Format | Promesse |
|:-------:|------|--------|----------|
| **I** | Theory of a Neuron | PDF 10 pages | Comprendre ce qu’est un neurone — sans code |
| **II** | The Art of Descent | PDF 11–12 pages | Comprendre les gradients et la chaîne de dérivation |
| **III** | Birth of a Neuron | PDF 18 pages + Colab | Coder le neurone de A à Z sur toxic vs non-toxic plants |

**Arc narratif :** Théorie → Mathématiques → Code.

---

## 2. Correspondance épisode par épisode

### Episode I — Theory of a Neuron

| Élément | Statut | Détail |
|---------|--------|--------|
| **Post LinkedIn** | ✅ | "NEURON" en commentaire → PDF envoyé en DM |
| **PDF** | ✅ Présent sur GitHub | `pdf/Theory of a Neuron.pdf` |
| **Contenu** | ✅ Aligné | Linear function, sigmoid, log-loss, gradient descent (sans détail des gradients) |
| **Notebook associé** | ⚠️ Partiel | Même thème que `01_single_neuron.ipynb`, mais Ep. I = théorie pure, pas de code |

**Conclusion :** Episode I = PDF uniquement. Le README lie Ep. 1 à `01_single_neuron.ipynb`, ce qui est cohérent dans l’esprit (thème neurone) mais Ep. I ne contient pas de code.

---

### Episode II — The Art of Descent

| Élément | Statut | Détail |
|---------|--------|--------|
| **Post LinkedIn** | ✅ | "GRADIENT" en commentaire → PDF envoyé en DM |
| **PDF** | ✅ Présent sur GitHub | `pdf/The Art of Descent.pdf` |
| **Contenu** | ✅ Aligné | Chain rule, ∂ℓ/∂a, ∂a/∂z, ∂z/∂w, ∂z/∂b, formule finale (a−y)xk |
| **Notebook associé** | ✅ Correspondant | `02_gradients_single_neuron.ipynb` (theory_02 / gradients) |

**Conclusion :** Episode II est bien relié au PDF et au notebook des gradients.

---

### Episode III — Birth of a Neuron

| Élément | Statut | Détail |
|---------|--------|--------|
| **Post LinkedIn** | ✅ | PDF + lien Colab en commentaire |
| **Colab** | ✅ Correct | `https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/birth_of_a_neuron.ipynb` |
| **Notebook** | ✅ Présent sur GitHub | `notebooks/birth_of_a_neuron.ipynb` |
| **PDF (Birth of a Neuron)** | ❌ Absent du repo | Le PDF 18 pages décrit dans le post n’est pas dans `pdf/` |

**Contenu du PDF Ep. III (résumé) :**
- Dataset : `make_blobs` (toxic vs non-toxic plants, 2 features)
- Boucle d’apprentissage : `initialisation` → `model` → `log_loss` → `gradients` → `update`
- Courbe d’apprentissage, `predict()`, `accuracy`, frontière de décision, prédiction sur un nouveau plant

**Conclusion :** Le PDF Ep. III est la **version texte** du notebook `birth_of_a_neuron.ipynb`, mais il n’est pas versionné sur GitHub.

---

## 3. Cartographie complète : LinkedIn ↔ GitHub

```
LinkedIn (3 épisodes)                    GitHub
─────────────────────────────────────────────────────────────────
Ep. I  Theory of a Neuron          →    pdf/Theory of a Neuron.pdf
      (PDF only)                         (+ thème proche de 01_single_neuron.ipynb)

Ep. II The Art of Descent          →    pdf/The Art of Descent.pdf
      (PDF only)                         notebooks/02_gradients_single_neuron.ipynb

Ep. III Birth of a Neuron          →    notebooks/birth_of_a_neuron.ipynb (Colab ✅)
      (PDF + Colab)                      pdf/Birth of a Neuron.pdf ✅
```

---

## 4. Ce qui existe sur GitHub vs ce que décrit le README

### 4.1 PDFs réels dans `pdf/`

| Fichier GitHub | Correspondance LinkedIn |
|----------------|-------------------------|
| `Theory of a Neuron.pdf` | Episode I |
| `The Art of Descent.pdf` | Episode II |
| `Birth of a Neuron.pdf` | Episode III ✅ (ajouté) |
| `main.pdf` | Non lié à la série 3 épisodes (probablement « capstone » global) |
| `mnist.pdf` | Série future (MNIST) |
| `CNN.pdf` | Série future (CNN) |

### 4.2 README actuel — Incohérences

Le README décrit une **autre série** (7 épisodes) qui ne correspond pas à ta série 3 épisodes :

| README dit | Réalité LinkedIn |
|------------|------------------|
| Ep. 1 : "I built a neuron from scratch" → `01_single_neuron` | Ep. 1 : Theory of a Neuron (PDF only) |
| Ep. 2 : "Log-loss explained" → `02_logloss_and_metrics` | Ep. 2 : The Art of Descent → `02_gradients_single_neuron` |
| Ep. 3 : "How backprop works" → `03_gradients_single_neuron` | Ep. 3 : Birth of a Neuron → `birth_of_a_neuron` + PDF manquant |
| Ep. 4–7 : Suite… | Pas encore publiés |

Le tableau LinkedIn du README mélange donc deux mondes :
1. Ta série actuelle (3 épisodes PDF/Colab)
2. Une série imaginée (7 épisodes, notebooks 01–12)

---

## 5. Liens pédagogiques entre contenus

### 5.1 Progression théorique

```
Ep. I (Theory of a Neuron)
├── z = wᵀx + b
├── a = σ(z)
├── Log-loss J(w,b)
└── Gradient descent (idée générale)

Ep. II (The Art of Descent)
├── Chain rule : ∂ℓ/∂w = ∂ℓ/∂a · ∂a/∂z · ∂z/∂w
├── Calcul explicite : ∂ℓ/∂wk = (a−y)xk, ∂ℓ/∂b = (a−y)
└── Mise à jour : w ← w − η∂J/∂w

Ep. III (Birth of a Neuron)
├── Code : initialisation, model, gradients, update
├── Boucle : model → loss → gradients → update
├── Courbe d’apprentissage, predict, accuracy
└── Frontière de décision, prédiction sur nouveau sample
```

### 5.2 Mapping notebooks ↔ épisodes

| Notebook GitHub | Rôle par rapport aux épisodes |
|-----------------|------------------------------|
| `01_single_neuron.ipynb` | Même thème qu’Ep. I, mais avec code (Ep. I = théorie seule) |
| `02_gradients_single_neuron.ipynb` | **Directement** lié à Ep. II |
| `birth_of_a_neuron.ipynb` | **Directement** lié à Ep. III (Colab dans le post) |

---

## 6. Synthèse des écarts

| # | Écart | Sévérité |
|---|-------|----------|
| 1 | **Birth of a Neuron.pdf** absent de `pdf/` | Haute — post LinkedIn renvoie vers un PDF qui n’est pas dans le repo |
| 2 | README : tableau LinkedIn 7 épisodes vs série réelle 3 épisodes | Moyenne — crée une confusion pour visiteurs et recruteurs |
| 3 | README : noms de PDFs (`main_capstone`, `mnist_guide`, `cnn_guide`) vs noms réels | Moyenne — liens cassés potentiels |
| 4 | README : notebooks 02, 03, 05, 06, 10 inexistants ou mal nommés | Moyenne — liens vers des fichiers absents |

---

## 7. Recommandations pour aligner GitHub et LinkedIn

### Priorité 1 — PDF Episode III
- [x] Ajouter `pdf/Birth of a Neuron.pdf` au dépôt (depuis episode 3.pdf dans Downloads). Fait.

### Priorité 2 — Section LinkedIn dans le README
- [x] Remplacer la section LinkedIn par une section dédiée aux 3 épisodes. Fait.
- [x] Indiquer clairement : PDF + Colab pour Ep. III. Fait.
- [x] Lier chaque épisode au bon PDF et au bon notebook. Fait.

### Priorité 3 — Structure PDF dans le README
- [x] Mettre à jour la table des guides avec les vrais noms. Fait.
- [x] Séparer : série 3 épisodes vs guides futurs. Fait.

### Priorité 4 — Cohérence narrative
- [x] Expliquer que les 3 épisodes = cours LinkedIn avec PDF. Fait.
- [x] Préciser que les notebooks = laboratoire GitHub. Fait.

---

## 8. Proposition de tableau README pour la série LinkedIn

```markdown
## 📢 LinkedIn Series — Deep Learning From Scratch (3 épisodes)

| Épisode | Titre | Contenu | Lien GitHub |
|:-------:|-------|---------|-------------|
| I | **Theory of a Neuron** | PDF 10 p. — linear function, sigmoid, log-loss | [`pdf/Theory of a Neuron.pdf`](pdf/Theory%20of%20a%20Neuron.pdf) |
| II | **The Art of Descent** | PDF 12 p. — chain rule, gradients ∂ℓ/∂w, ∂ℓ/∂b | [`pdf/The Art of Descent.pdf`](pdf/The%20Art%20of%20Descent.pdf) · [`02_gradients_single_neuron.ipynb`](notebooks/02_gradients_single_neuron.ipynb) |
| III | **Birth of a Neuron** | PDF 18 p. + Colab — neurone codé à la main sur toxic plants | [`pdf/Birth of a Neuron.pdf`](pdf/Birth%20of%20a%20Neuron.pdf) · [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/notebooks/birth_of_a_neuron.ipynb) |

Comment "NEURON" (Ep. I), "GRADIENT" (Ep. II) ou "BIRTH" (Ep. III) sur LinkedIn pour recevoir le PDF en DM.
```

---

## 9. Vue d’ensemble


**Mise à jour (27 fév. 2026) :** Toutes les recommandations ont été mises en œuvre : PDF Ep. III ajouté, README mis à jour, requirements.txt créé.

Ta série LinkedIn a une **structure pédagogique claire** (theory → gradients → code). Les 3 épisodes sont maintenant alignés entre LinkedIn et GitHub.
