# Data

This project uses two sessions from the Perich & Miller (2018) center-out reaching dataset, recorded from monkey "T":

| Session ID                              | Role                  | Size    |
| :-------------------------------------- | :-------------------- | :------ |
| `t_20130819_center_out_reaching.h5`     | Day 1 (pre-training)  | ~9.9 MB |
| `t_20130821_center_out_reaching.h5`     | Day 2 (adapt + test)  | ~10.4 MB |

The `.h5` files are not committed to this repo. Fetch them with the commands below before running the notebook.

## How to download

The notebook (`code/neural_decoding.ipynb`) downloads the files automatically. To fetch them manually:

```bash
pip install gdown
mkdir -p data/perich_miller_population_2018
gdown 1W--Sm_BcphEC2snoF4zwPdHkkYGgAaUw \
  -O data/perich_miller_population_2018/t_20130819_center_out_reaching.h5
gdown 1EZUkOE8oiieWja9lblokf8WjF05HFuX- \
  -O data/perich_miller_population_2018/t_20130821_center_out_reaching.h5
```

The notebook expects this exact directory layout (`data/perich_miller_population_2018/<session>.h5`).

## Source

Perich, M. G., Gallego, J. A., & Miller, L. E. (2018). *A neural population mechanism for rapid learning.* Neuron 100(4), 964-976. The data is hosted on Google Drive by the `torch_brain` maintainers; see [`torch_brain`](https://github.com/neuro-galaxy/torch_brain) for additional sessions and details.
