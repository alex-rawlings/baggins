#!/bin/bash -l
#SBATCH --job-name=analysis
#SBATCH --account=pjohanss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=16:00:00
#SBATCH --partition=medium
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py393

#python ./bh_perturb_distribution_new.py "/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/output" -l 51

python extract_HM_quantities.py "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/AD/AD-030-0050.py" "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.py"

python extract_HM_quantities.py "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/AD/AD-030-0010.py" "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.py"