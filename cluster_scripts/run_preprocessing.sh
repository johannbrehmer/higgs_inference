#!/bin/bash

#SBATCH --job-name=preprocessing
#SBATCH --output=slurm_preprocessing_small.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=8:00:00

# Modules
module purge
module load jupyter-kernels/py2.7
module load scikit-learn/intel/0.18.1

cd /home/jb6504/higgs_inference/preprocessing

#python -u generate_data.py --train --random --basis --pointbypoint --scoreregression --new
python -u generate_data.py --neyman --dry
