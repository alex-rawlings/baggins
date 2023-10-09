#!/bin/bash -l
#SBATCH --job-name=HMQ
#SBATCH --account=pjohanss
#SBATCH --ntasks-per-node=16
#SBATCH --time=168:00:00
#SBATCH --partition=interactive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py393

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/core-study/e-097/hernquist_2M_bc.yml -m kick -k /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/eccentricity-study/e-099/dehnen_1M_0005BH_mergers -m mc

cd hierarchical_models


#python quinlan_hardening.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/eccentricity_study/D_4M-D_4M-3.720-0.279/ new -v DEBUG

#python quinlan_hardening.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/hardening/mcs/D_100K-D_100K-3.720-0.279/quinlan_hierarchy-20230704143650_*.csv" loaded -v DEBUG

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/core_study/H_2M-H_2M-30.000-2.000/ new -m factor
