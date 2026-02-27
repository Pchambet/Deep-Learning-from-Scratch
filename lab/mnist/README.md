# MNIST Case Study

MLP sur le dataset MNIST — pipeline complet, de la normalisation au training.

## Lancer

**Notebook :**
```bash
jupyter notebook mnist.ipynb
```
Ou [Colab](https://colab.research.google.com/github/Pchambet/Deep-Learning-from-Scratch/blob/main/lab/mnist/mnist.ipynb)

**Script :**
```bash
make mlp
```
Sauvegarde dans `outputs/`.

## Dépendances

Utilisez le `requirements.txt` à la racine du projet, ou :
```bash
pip install tensorflow numpy matplotlib
```
Sur Apple Silicon : `tensorflow-macos` + `tensorflow-metal`.

## Dataset

MNIST est téléchargé automatiquement via `tensorflow.keras.datasets.mnist`.
