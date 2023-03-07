#!/bin/bash
#SBATCH -n 40
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --constraint=32GB
#SBATCH -t 24:00:00
#SBATCH -w dgx001

source ~/.bashrc
cd /om2/user/gua/Documents/CodeRanker/src
wandb enabled
./run_codebert2.sh
