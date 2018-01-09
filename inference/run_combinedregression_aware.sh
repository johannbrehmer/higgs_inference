#!/bin/bash

#SBATCH --job-name=creg-aware
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

# Modules
module purge
module load jupyter-kernels/py2.7
module load scikit-learn/intel/0.18.1
module load theano/0.9.0
module load tensorflow/python2.7/20170707
module load keras/2.0.2

cd /home/jb6504/learning_higgs_eft/parameterized/inference/cluster

python -u parameterized_inference.py combinedregression aware basis shallow
python -u parameterized_inference.py combinedregression aware basis
python -u parameterized_inference.py combinedregression aware basis deep

