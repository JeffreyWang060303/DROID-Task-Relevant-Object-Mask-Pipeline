#!/bin/bash
#SBATCH --job-name=objmask
#SBATCH --partition=viscam
#SBATCH --account=viscam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate sam3

OFFSET=$SLURM_ARRAY_TASK_ID
COUNT=1
EXP_ID=$((OFFSET + 1))

python task_relevant_object_mask_pipeline.py \
  --offset ${OFFSET} \
  --count ${COUNT}
