#!/bin/bash -l
#SBATCH --job-name=autoregress
#SBATCH --account=pjohanss
#SBATCH --time=24:00:00
#SBATCH --cores=8
#SBATCH --partition=interactive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py393

python autoregress.py ../data/all_data_10pc.pickle inv_a