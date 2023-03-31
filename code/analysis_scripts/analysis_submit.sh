#!/bin/bash -l
#SBATCH --job-name=analysisH
#SBATCH --account=pjohanss
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00
#SBATCH --partition=interactive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py393

cd hierarchical_models

python quinlan_hardening.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes_new/MC_sample/H_1-000/ /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/hernquists_mcs/H_1-000/ -m mc 

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/hernquists_mcs/H_0-500/ -m mc 

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/hernquists_mcs/H_0-250/ -m mc 

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/hernquists_mcs/H_0-100/ -m mc 

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/hernquists_mcs/H_0-050/ -m mc 

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/hernquists_mcs/H_0-025/ -m mc 

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/hernquists_mcs/H_0-010/ -m mc 

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/hernquists_mcs/H_0-005/ -m mc -o
