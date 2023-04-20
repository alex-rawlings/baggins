#!/bin/bash -l
#SBATCH --job-name=ana_HMQ
#SBATCH --account=pjohanss
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00
#SBATCH --partition=interactive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py393

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/nasim/SO_e_high/hernquist_HR_SO/ -m mc

cd hierarchical_models

python quinlan_hardening.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes_new/nasim/stars_only_e_high/HR /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml

python quinlan_hardening.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes_new/nasim/stars_only_e_high/MR /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml

python quinlan_hardening.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes_new/nasim/stars_only_e_high/LR /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml

python quinlan_hardening.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes_new/nasim/stars_only_e_high/PR /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml
