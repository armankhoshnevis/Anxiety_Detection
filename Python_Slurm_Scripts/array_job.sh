#!/bin/bash --login
#SBATCH --job-name=SVC
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=../Results/SVC/logs/%x_%A_%a.out
#SBATCH --error=../Results/SVC/logs/%x_%A_%a.err

# Exit on error, undefined variable, or error in a pipeline
set -euo pipefail

# Load modules and Python environment
module purge
module load Miniforge3/24.3.0-0
conda activate Voice_Project

# Run ONE scenario per array task
python Train_ML.py \
  --model_name SVC \
  --n_repeats 7 \
  --outer_splits 5 \
  --inner_splits 5 \
  --n_iter 10000 \
  --verbose 1 \
  --outer_n_jobs "$SLURM_CPUS_PER_TASK"\
  --inner_n_jobs 1
