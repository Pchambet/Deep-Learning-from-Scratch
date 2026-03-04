# Fondations saines pour l'Episode V — From One Neuron to a Brain

**Objectif** : Poser les bases pour construire l'épisode 5 de manière alignée avec les épisodes 1–4, en capitalisant sur ce qui existe déjà.

---

## 1. Ce que dit la fin de l'Episode IV

> « Let's move from a neuron... to a brain. »  
> « We add more neurons. We add more layers. We move from a single-cell brain to a full Artificial Neural Network (ANN). »  
> « In the next episode, we build our first real neural network. »

→ **Episode V = premier vrai réseau de neurones** (au moins 2 couches).

---

## 2. Ce qui existe déjà dans le repo

| Fichier | Contenu | Utilisable pour Ep. V ? |
|---------|---------|--------------------------|
| `notebooks/08_two_layer_network.ipynb` | Réseau 2 couches : plants (make_blobs/circles) puis cats vs dogs. Init, forward, backprop, update, predict. | ✅ Base principale |
| `notebooks/07_two_layer_gradients.ipynb` | Détail des gradients / backprop pour 2 couches | ✅ Référence pour la partie théorie |
| `notebooks/09_backprop_any_depth.ipynb` | Généralisation à L couches | ⏭ Plus tard (Ep. VI ?) |
| `src/utilities.py` | `load_data()` pour cats/dogs | ✅ Réutilisable |
| `notebooks/birth_of_a_neuron.py` | Neurone 1 couche | ❌ Ep. V = 2 couches, nouveau module |

---

## 3. Arc narratif proposé pour l'Episode V

```
Ep. IV a montré : 1 neurone → 1 droite → limite sur images
Ep. V : on ajoute une couche cachée
  → Forward : X → Z1 → A1 → Z2 → A2 (prediction)
  → Backprop : mêmes idées (chaîne) mais en chaîne de matrices
  → Résultat : frontière de décision non linéaire
  → Même dataset cats vs dogs (continuité) ou make_circles (visuel)
```

**Deux options pour le dataset :**

| Option | Avantage | Inconvénient |
|--------|----------|--------------|
| **A — make_circles / make_blobs** | Visuel (2D), montre clairement que 1 neurone échoue et 2 couches réussissent | Change de thème (plus de cats/dogs) |
| **B — Cats vs dogs (4096)** | Continuité directe avec Ep. IV | Plus lourd, moins visuel |

**Recommandation** : Option A pour le **PDF** (make_circles : 1 neurone vs 2 couches, comparaison visuelle). Le **notebook** peut ensuite enchaîner sur cats/dogs comme dans `08_two_layer_network` (section existante).

---

## 4. Structure technique à mettre en place

### 4.1 Module Python réutilisable

Créer `notebooks/two_layer_network.py` (ou `src/two_layer_network.py`) avec :

- `initialisation(n0, n1, n2)`  
- `forward_propagation(X, parameters)`  
- `backward_propagation(X, y, activations, parameters)`  
- `update_parameters(parameters, gradients, learning_rate)`  
- `predict(X, parameters)`  
- `neural_network(X, y, n1=32, learning_rate=0.1, epoch=1000)` (boucle d’entraînement)

→ Même logique que `birth_of_a_neuron.py` pour Ep. III : le notebook importe, Colab peut cloner le repo.

### 4.2 Notebook Episode V

**Deux approches :**

1. **Adapter `08_two_layer_network.ipynb`**  
   - Renommer / créer une copie `05_from_neuron_to_brain.ipynb`  
   - Mettre à jour le titre, l’intro, le setup Colab (comme pour 04)  
   - Séquencer : plants (make_circles) → cats/dogs (optionnel)

2. **Nouveau notebook dédié**  
   - `05_neural_network_from_scratch.ipynb`  
   - Contenu ciblé : uniquement l’essentiel (init, forward, backprop, update, courbes)  
   - Imports depuis `two_layer_network.py`

→ **Conseil** : Option 1 (adapter 08) pour aller plus vite. Option 2 si tu veux un notebook plus court et mieux aligné sur le PDF.

### 4.3 Setup Colab

Reprendre le même pattern que `04_training_loop_from_scratch` :

- Détection Colab  
- Clone du repo si nécessaire  
- `load_data()` pour cats/dogs  
- `make_circles` / `make_blobs` si utilisé  

---

## 5. Structure LaTeX (PDF)

Créer `latex/episode_05/` sur le modèle d’`episode_04` :

```
latex/episode_05/
├── episode_05.tex
├── Makefile
└── (assets si besoin)
```

**Plan du PDF :**

1. Page de titre + lien Colab  
2. **Rappel Ep. IV** : limite du neurone unique, une droite  
3. **Make_circles** (ou problème 2D) : 1 neurone échoue, 2 couches réussissent  
4. **Architecture** : input → couche cachée → output, dimensions (n0, n1, n2)  
5. **Initialisation** : W1, b1, W2, b2, pourquoi les dimensions  
6. **Forward propagation** : Z1, A1, Z2, A2, formules matricielles  
7. **Backpropagation** : idée (chaîne), formules clés sans tout détailler  
8. **Boucle d’entraînement** : forward → loss → backward → update (comme Ep. III/IV)  
9. **Résultat** : frontière de décision non linéaire, comparaison 1 vs 2 couches  
10. **Conclusion** : "On a construit un vrai réseau. Prochaine étape : plus de couches / MNIST / CNN."

---

## 6. Checklist avant de démarrer l’Episode V

### À faire en priorité

- [ ] Extraire les fonctions de `08_two_layer_network` dans `two_layer_network.py`  
- [ ] Vérifier que le notebook 08 fonctionne sur Colab (clone repo, imports)  
- [ ] Créer `latex/episode_05/` avec un squelette minimal  

### À faire ensuite

- [ ] Rédiger le PDF (épisode_05.tex) en suivant le plan ci-dessus  
- [ ] Créer ou adapter le notebook `05_*.ipynb` avec titre "From One Neuron to a Brain"  
- [ ] Mettre à jour le README (ajout Ep. V dans la série LinkedIn)  

### Optionnel

- [ ] Générer des figures (frontière 1 neurone vs 2 couches sur make_circles)  
- [ ] Préparer une release v0.1-data si les HDF5 cats/dogs ne sont pas déjà disponibles  

---

## 7. Résumé — ordre des étapes

1. **Extraire `two_layer_network.py`** depuis le notebook 08  
2. **Tester** le notebook 08 sur Colab (avec setup style Ep. 4)  
3. **Créer** `latex/episode_05/episode_05.tex` avec le plan narratif  
4. **Rédiger** le PDF section par section  
5. **Créer ou adapter** le notebook Ep. V et le lier au PDF  
6. **Mettre à jour** le README et préparer le post LinkedIn  

Tu peux commencer par l’étape 1 (extraction de `two_layer_network.py`) puis enchaîner avec le setup Colab du notebook 08.
