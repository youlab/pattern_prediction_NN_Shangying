#!/bin/bash
#SBATCH -o slurmj1.out
#SBATCH -e slurmj1.err
#SBATCH -p gpu-common --gres=gpu:1 
#SBATCH --mem=20G 
#SBATCH -c 6 
python mymodel_training.py
