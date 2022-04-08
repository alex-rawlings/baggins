#!/bin/bash -l
#SBATCH --job-name=adding
#SBATCH --account=pjohanss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:00:00
#SBATCH --partition=test

module purge
module restore py386
source "/users/arawling/py386venv/bin/activate"

python add_field.py
