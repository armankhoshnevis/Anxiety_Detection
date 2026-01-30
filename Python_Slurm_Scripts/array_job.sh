#!/bin/bash --login
#SBATCH --job-name=SVC
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
#SBATCH --time=2:00:00
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
  --n_repeats 3 \
  --outer_splits 3 \
  --inner_splits 3 \
  --n_iter 100 \
  --verbose 1 \
  --outer_n_jobs "$SLURM_CPUS_PER_TASK"\
  --inner_n_jobs 1
