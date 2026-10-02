# Data: cat vs dog, 64×64 grayscale

| File | Images | Classes | Shape | Keys |
| --- | --- | --- | --- | --- |
| `trainset.hdf5` | 1,000 | 500 cats (0), 500 dogs (1) | `(1000, 64, 64)` uint8 | `X_train`, `Y_train` |
| `testset.hdf5` | 200 | 100 cats (0), 100 dogs (1) | `(200, 64, 64)` uint8 | `X_test`, `Y_test` |

Used by Episodes IV–VII and by `scripts/make_figures.py`. Load them with
`src.utilities.load_data()`; the class balance is checked in `tests/test_utilities.py`.

The files are small (4.9 MB) and committed, so a clone is self-contained. On Colab the
notebooks download them from this folder if they are missing. `load_data()` raises
`FileNotFoundError` when a file is absent, so no script ever reports results on placeholder data.

Origin: a small teaching set of photographs reduced to 64×64 grayscale. Its original
source and licence are not recorded in this repository; it is used here for teaching only.
