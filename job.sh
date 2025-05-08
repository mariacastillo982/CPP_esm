#!/bin/bash

#SBATCH --job-name=cpp
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL

module purge
#module load openfold/1.0.1
module load cuda/11.7.1

python pLM_graph.py
