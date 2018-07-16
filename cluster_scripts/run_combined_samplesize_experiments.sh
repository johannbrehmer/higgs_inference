#!/bin/bash

#SBATCH --job-name=comb-size2
#SBATCH --output=slurm_combined_samplesize2.out
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

# python -u experiments.py combined --samplesize 1000 -o deep
# python -u experiments.py combined --samplesize 2000 -o deep
#python -u experiments.py combined --samplesize 5000 -o deep
#python -u experiments.py combined --samplesize 10000 -o deep
#python -u experiments.py combined --samplesize 20000 -o deep
#python -u experiments.py combined --samplesize 50000 -o deep
python -u experiments.py combined --samplesize 100000 -o deep  neyman2 --neyman
#python -u experiments.py combined --samplesize 200000 -o deep
#python -u experiments.py combined --samplesize 500000 -o deep
#python -u experiments.py combined --samplesize 1000000 -o deep
#python -u experiments.py combined --samplesize 2000000 -o deep
#python -u experiments.py combined --samplesize 5000000 -o deep
