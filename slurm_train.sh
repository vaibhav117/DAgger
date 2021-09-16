#!/bin/bash -e
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=hw1_expert_explore
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --array=1
#SBATCH --output=./logs/expert_explore/%j_%x.out
#SBATCH --error=./logs/expert_explore/%j_%x.err
#SBATCH --export=ALL

bash ./run_singularity_mpi.sh \
	python3 proto_goal/test_asym_actor_critic.py
