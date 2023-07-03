#!/bin/bash -l
#SBATCH --job-name=ana_HMQ
#SBATCH --account=pjohanss
#SBATCH --ntasks-per-node=128
#SBATCH --time=10:00:00
#SBATCH --partition=medium
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py393


python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/eccentricity-study/e-090/dehnen_8M_mergers -m mc

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/eccentricity-study/e-099/dehnen_2M_mergers -m mc


#cd hierarchical_models

#python quinlan_hardening.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes_new/nasim/stars_only_e_high/HR /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml

#python binary_properties.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes_new/MC_sample/with_energy/H_1-000 /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml
