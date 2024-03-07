#!/bin/bash -l
#SBATCH --job-name=HMQ
#SBATCH --account=pjohanss
#SBATCH --ntasks-per-node=16
#SBATCH --time=04:00:00
#SBATCH --partition=interactive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py393

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/core-study/e-097/hernquist_2M_bc.yml -m kick -k /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml

#python extract_HM_quantities.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml /users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/eccentricity-study/e-099/dehnen_1M_0005BH_mergers -m mc

cd hierarchical_models

echo "Prior sensitivity analysis"

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0000/graham_hierarchy-20231126160359_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0060/graham_hierarchy-20231128225655_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0120/graham_hierarchy-20231129100234_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0180/graham_hierarchy-20231130191605_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0240/graham_hierarchy-20231201053636_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0300/graham_hierarchy-20231201152531_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0360/graham_hierarchy-20231201224632_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0420/graham_hierarchy-20231202095302_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0480/graham_hierarchy-20231203200318_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0540/graham_hierarchy-20231204014034_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0600/graham_hierarchy-20231127184417_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0660/graham_hierarchy-20231204110157_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0720/graham_hierarchy-20231204195448_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0780/graham_hierarchy-20231205010214_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0840/graham_hierarchy-20231205072236_*.csv" loaded

python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/mcs/H_2M-H_2M-30.000-2.000-v0900/graham_hierarchy-20231128004407_*.csv" loaded

