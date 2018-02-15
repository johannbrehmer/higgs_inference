#!/bin/bash

#SBATCH --job-name=carl-learning2
#SBATCH --output=slurm_carl_learning_experiments2.out
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

cd /home/jb6504/higgs_inference/higgs_inference

python -u experiments.py carl -o largebatch constantlr
python -u experiments.py carl -o largebatch fastlearning constantlr
python -u experiments.py carl -o largebatch slowlearning constantlr

python -u experiments.py carl -o constantlr
python -u experiments.py carl -o fastlearning constantlr
python -u experiments.py carl -o slowlearning constantlr

python -u experiments.py carl -o smallbatch constantlr
python -u experiments.py carl -o smallbatch fastlearning constantlr
python -u experiments.py carl -o smallbatch slowlearning constantlr
