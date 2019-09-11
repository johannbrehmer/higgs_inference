# Code repository for the paper "Constraining Effective Field Theories with Machine Learning"

Johann Brehmer, Kyle Cranmer, Gilles Louppe, and Juan Pavez

**Note**: If you want to use these methods for a particle physics problem, please have a look at our new tool [MadMiner](https://github.com/diana-hep/madminer) -- you will likely find that much easier to use than this repo!

## Folder structure

- `cluster_scripts`: SLURM scripts that start the preprocessing and experiments on the NYU HPC cluster.
- `data`: The data set, including both the original weighted event sample as well as the unweighted training, calibration, and evaluation samples. Some of these are quite large and not on GitHub.
- `evaluation`: IPython notebooks that extract metrics and figures from the experiments.
- `figures`: Here the figures are stored.
- `higgs_inference`: The main folder for the inference experiments.
    - `models`: The keras model code at the heart of many inference strategies.
    - `strategies`: Training and evaluation routines for the different inference strategies.
    - `various`: Different utility functions.
    - `experiments.py`: Main executable that starts the different training and evaluation pieces.
    - `settings.py`:  Most settings and constants, including the main directories, event numbers, architecture parameters, and benchmark thetas.
- `preprocessing`: Unweighting routines that generate the different training, calibration, and evaluation samples from the original weighted event file.
- `postprocessing`: Code for the Neyman construction.
- `prototypes`: Toy experiments and cross-checks on other data. Includes the `flow` folder with a PyTorch implementation of normalizing flows.
- `results`: The predictions of the different algorithms on the test data set.
