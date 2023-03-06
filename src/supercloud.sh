#!/bin/bash

#SBATCH --gres=gpu:volta:2
#SBATCH -c 40
#SBATCH -N 1
#SBATCH --exclusive

module load anaconda/2022b
module load cuda/11.6

source activate coderanker

export HF_DATASETS_OFFLINE=1
export HF_HOME=/state/partition1/user/agu/.cache/huggingface
export TMPDIR=/state/partition1/user/agu
                                            
cd /home/gridsan/agu/Documents/CodeRanker/src
wandb offline
export CUDA_VISIBLE_DEVICES=0,1
bash finetune_partial.sh ../t5 ../models_t5 ../cache_t5 binary