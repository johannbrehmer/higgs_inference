#!/bin/bash

#SBATCH --job-name=higgs_maf_samplesize
#SBATCH --output=slurm_maf_samplesize.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

# Modules
#module purge
#module load jupyter-kernels/py2.7
#module load scikit-learn/intel/0.18.1
#module load theano/0.9.0
#module load tensorflow/python2.7/20170707
#module load keras/2.0.2

source activate goldmine
cd /home/jb6504/higgs_inference/higgs_inference

python -u experiments.py maf -o deep --samplesize 1000
python -u experiments.py maf -o deep --samplesize 2000
python -u experiments.py maf -o deep --samplesize 5000
python -u experiments.py maf -o deep --samplesize 10000
python -u experiments.py maf -o deep --samplesize 20000
python -u experiments.py maf -o deep --samplesize 50000
python -u experiments.py maf -o deep --samplesize 100000
python -u experiments.py maf -o deep --samplesize 200000
python -u experiments.py maf -o deep --samplesize 500000
python -u experiments.py maf -o deep --samplesize 1000000
python -u experiments.py maf -o deep --samplesize 2000000
python -u experiments.py maf -o deep --samplesize 5000000
