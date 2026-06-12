# Problèmes identifiés et corrections — Synthèse complète

**Date :** 10 mars 2026  
**Objectif :** Liste exhaustive des problèmes identifiés dans les audits (général + Episode 6) et statut de correction.

---

## 1. Audit général du projet (session précédente)

| # | Problème | Fichier | Correction | Statut |
|---|----------|---------|------------|--------|
| 1 | CI échoue sans HDF5 (load_data sans fallback) | `src/utilities.py` | Ajout `_synthetic_data()` + fallback dans `load_data()` | ✅ |
| 2 | README référence notebook 09 inexistant | `README.md` | Suppression référence, correction table | ✅ |
| 3 | `scripts/` exclu de Ruff (pre-commit) | `.pre-commit-config.yaml` | Ajout `scripts/` au pattern | ✅ |
| 4 | Duplication `set_seed` + courbes (train_mlp, train_cnn) | `lab/` | Création `lab/common.py`, factorisation | ✅ |
| 5 | `smoke_test` réimplémente le neurone | `scripts/smoke_test.py` | Import `birth_of_a_neuron`, fix broadcast y | ✅ |
| 6 | `run_04_test` réimplémente le neurone | `notebooks/run_04_test.py` | Import `birth_of_a_neuron` | ✅ |
| 7 | Pas de tests unitaires | — | Suite pytest (22 tests) | ✅ |
| 8 | CI sans pytest | `.github/workflows/ci.yml` | Étape pytest ajoutée | ✅ |
| 9 | Makefile PYTHON sans venv | `Makefile` | Préférence `.venv/bin/python` si présent | ✅ |

---

## 2. Audit Episode 6 — `06_alive.ipynb`

| # | Problème | Cellule | Correction | Statut |
|---|----------|---------|------------|--------|
| 10 | `y_train[i]` → risque ambiguïté NumPy | 28 | `y_train.flat[i]` | ✅ |
| 11 | « Exact equations from Episode V » — Ep. V = tanh, Ep. VI = sigmoid | 7, 9, 11 | Clarification : sigmoid partout (simplification), Ep. V utilise tanh | ✅ |
| 12 | 6 entraînements sans indication temps | 24 | Note « ≈ 2–3 min » ajoutée | ✅ |
| 13 | 10000 epochs sans indication temps | 31 | Note « ≈ 1–2 min » ajoutée | ✅ |

---

## 3. Synthèse

| Catégorie | Problèmes | Corrigés |
|-----------|-----------|----------|
| Projet (infra, CI, code) | 9 | 9 |
| Episode 6 (notebook) | 4 | 4 |
| **Total** | **13** | **13** |

---

*Document généré après audit complet et corrections — 10 mars 2026*
