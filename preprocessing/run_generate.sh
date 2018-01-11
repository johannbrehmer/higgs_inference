#!/bin/bash

#SBATCH --job-name=data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=255GB
#SBATCH --time=8:00:00

# Modules
module purge
module load jupyter-kernels/py2.7
module load scikit-learn/intel/0.18.1

cd /home/jb6504/higgs_inference/preprocessing
#python -u generate_data.py train
#python -u generate_data.py basis
#python -u generate_data.py random
python -u generate_data.py point-by-point
#python -u generate_data.py calibration
#python -u generate_data.py test
#python -u generate_data.py score-regression
