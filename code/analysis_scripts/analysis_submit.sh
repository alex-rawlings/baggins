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

vkick=("0000" "0060" "0120" "0180" "0240" "0300" "0360" "0420" "0480" "0540" "0600" "0660" "0720" "0780" "0840" "0900" "0960" "1020" "1080" "1140" "1200" "1260" "1320" "1380" "1440" "1500" "1560" "1620" "1680" "1740" "1800" "2000")

cd hierarchical_models

for v in ${vkick[@]}; do
    python graham_density.py /users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/core_study/H_2M-H_2M-30.000-2.000/HMQ-cube-H_2M_b-H_2M_c-30.000-2.000-v$v.hdf5" new -m hierarchy
    echo
    echo
done