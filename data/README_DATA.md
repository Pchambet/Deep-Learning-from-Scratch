# Data Folder

This folder contains the training and test datasets used in the image classification experiments (Cat vs Dog).

- **trainset.hdf5** — Training data (1000 images 64×64, labels 0/1)
- **testset.hdf5** — Test data (200 images 64×64, labels 0/1)

## Sur Colab ou sans les fichiers locaux

Les fichiers HDF5 sont dans `.gitignore` (trop volumineux pour le dépôt). Le notebook **04_training_loop_from_scratch** gère automatiquement deux cas :

1. **Données présentes** (clone local avec `data/`) → utilisation normale.
2. **Données absentes** (Colab après clone GitHub) → génération de **données de démo synthétiques** (même format), le notebook s’exécute entièrement.

## Téléchargement optionnel depuis GitHub Releases

Pour utiliser les vraies images cat vs dog sur Colab, créez une release `v0.1-data` et attachez `trainset.hdf5` et `testset.hdf5`. Le notebook tentera alors de les télécharger automatiquement.
