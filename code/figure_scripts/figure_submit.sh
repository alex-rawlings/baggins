#!/bin/bash -l
#SBATCH --job-name=figure
#SBATCH --account=pjohanss
#SBATCH --time=36:00:00
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py386

python shell_motions.py -n