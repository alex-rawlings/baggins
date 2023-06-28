#!/bin/bash -l
#SBATCH --job-name=quinlan
#SBATCH --account=pjohanss
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00
#SBATCH --partition=interactive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py393

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/eccentricity-study/e-099/dehnen_2M_mergers -m mc


cd hierarchical_models

python quinlan_hardening.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_100K-D_100K-3.720-0.279 new
