#!/bin/bash

#SBATCH --job-name=svm_train
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50G
#SBATCH --time=26:00:00
#SBATCH --chdir=/scratch/s5130727/Applied-ML
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Informative messages for the log
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Initial working directory (set by --chdir): $(pwd)"

# Load necessary modules
echo "Loading Anaconda module..."
module load Anaconda3

# Activate Conda environment
echo "Activating Conda environment AML..."
conda activate AML

# Execute the Python script
# Using 'python' interpreter for the .py file
echo "Executing Python script: python -m src.models.modular_svm"
python -m src.models.modular_svm

echo "Python script finished."
echo "Job finished at: $(date)"
