#!/bin/bash
#SBATCH -A p32748  # Account name
#SBATCH -p gengpu  # GPU partition
#SBATCH --gres=gpu:a100:1  # Request 1 A100 GPU
#SBATCH -N 1  # Number of nodes
#SBATCH -n 20  # Number of tasks
#SBATCH -t 8:00:00  # Max runtime
#SBATCH --mem=32G  # Memory allocation
#SBATCH --job-name=train_coeff  # Job name
#SBATCH --output=train_%j.log  # Log file (SLURM_JOB_ID included)

# Load Python module
module load python/3.12.10

# Ensure real-time logging
export PYTHONUNBUFFERED=1

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment '.venv' not found!"
    exit 1
fi

# Run training script and log output
time python train_afhqv2.py