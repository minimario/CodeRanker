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
bash finetune_partial.sh ../t5_full_ternary ../models_t5_full_ternary ../cache_t5_full_ternary ternary /home/gridsan/agu/Documents/apps/train_full_introductory_dedup