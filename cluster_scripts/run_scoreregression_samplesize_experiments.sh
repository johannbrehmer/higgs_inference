#!/bin/bash

#SBATCH --job-name=sreg-size
#SBATCH --output=slurm_scoreregression_samplesize.out
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

#python -u experiments.py scoreregression --samplesize 1000 -o deep
#python -u experiments.py scoreregression --samplesize 2000 -o deep
#python -u experiments.py scoreregression --samplesize 5000 -o deep
#python -u experiments.py scoreregression --samplesize 10000 -o deep
#python -u experiments.py scoreregression --samplesize 20000 -o deep
#python -u experiments.py scoreregression --samplesize 50000 -o deep
python -u experiments.py scoreregression --samplesize 100000 -o deep neyman2 --neyman
#python -u experiments.py scoreregression --samplesize 200000 -o deep
#python -u experiments.py scoreregression --samplesize 500000 -o deep
#python -u experiments.py scoreregression --samplesize 1000000 -o deep
#python -u experiments.py scoreregression --samplesize 2000000 -o deep
#python -u experiments.py scoreregression --samplesize 5000000 -o deep
