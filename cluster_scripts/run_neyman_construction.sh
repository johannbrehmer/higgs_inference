#!/bin/bash

#SBATCH --job-name=nc
#SBATCH --output=slurm_neyman_construction2.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=62GB
#SBATCH --time=24:00:00

# Modules
module purge
module load jupyter-kernels/py2.7
module load scikit-learn/intel/0.18.1

cd /home/jb6504/higgs_inference/postprocessing

python -u neyman_construction.py --mxe --combined --combinedregression --combinedmxe --samplesize 100000 --set 2
