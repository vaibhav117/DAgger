#!/bin/bash -e
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=exp_highres
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --array=1
#SBATCH --output=./logs/dagger_explore_exponential_decay_highres/%j_%x.out
#SBATCH --error=./logs/dagger_explore_exponential_decay_highres/%j_%x.err
#SBATCH --export=ALL

bash ./run-mpi4py-singularity.bash \
	python3 dagger_template.py
