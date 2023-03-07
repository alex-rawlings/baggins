#!/bin/bash -l
#SBATCH --job-name=analysisH
#SBATCH --account=pjohanss
#SBATCH --ntasks-per-node=16
#SBATCH --time=20:00:00
#SBATCH --partition=interactive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py393

cd hierarchical_models

python quinlan_hardening.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H-H-3.0-0.001/ /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml

#python binary_properties.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H-H-3.0-0.001/ /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml -m hierarchy

#python graham_density.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H-H-3.0-0.001 /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml -m "hierarchy"

#python graham_density.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H0500-H0500-3.0-0.001 /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml 

#python graham_density.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H0250-H0250-3.0-0.001 /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml 

#python graham_density.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H0100-H0100-3.0-0.001 /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml 

#python graham_density.py /scratch/pjohanss/arawling/collisionless_merger/mergers/HMQcubes/H0050-H0050-3.0-0.001 /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml 