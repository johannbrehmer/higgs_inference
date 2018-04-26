# Code repository for the paper "Constraining Effective Field Theories with Machine Learning"

Johann Brehmer, Kyle Cranmer, Gilles Louppe, and Juan Pavez

## Folder structure

- `cluster_scripts`: SLURM scripts that start the preprocessing and experiments.
- `data`: The data set, including both the original weighted event sample as well as the unweighted training, calibration, and evaluation samples. Some of these are quite large and not on GitHub.
- `evaluation`: IPython notebooks that extract metrics and figures from the experiments.
- `figures`: Here the figures are stored.
- `higgs_inference`: The main folder for the inference experiments.
    - `models`: The keras model code at the heart of many inference strategies.
    - `strategies`: Training and evaluation routines for the different inference strategies.
    - `various`: Different utility functions and the p-value calculation.
    - `experiments.py`: Main executable that starts the different training and evaluation pieces.
    - `settings.py`:  Most settings and constants, including the main directories, event numbers, architecture parameters, and benchmark thetas.
- `preprocessing`: Unweighting routines that generate the different training, calibration, and evaluation samples from the original weighted event file.
- `prototypes`: Toy experiments and cross-checks on other data. Includes the `flow` folder with a PyTorch implementation of normalizing flows.
- `results`: The predictions of the different algorithms on the test data set.
