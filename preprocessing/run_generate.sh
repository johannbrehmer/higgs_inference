#!/bin/bash

#SBATCH --job-name=data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB
#SBATCH --time=48:00:00

# Modules
module purge
module load jupyter-kernels/py2.7
module load scikit-learn/intel/0.18.1

cd /home/jb6504/learning_higgs_eft/parameterized/inference/cluster
python -u generate_data.py basis