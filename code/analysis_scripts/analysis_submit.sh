#!/bin/bash -l
#SBATCH --job-name=analysis
#SBATCH --account=pjohanss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --partition=medium
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py386
source "/users/arawling/py386venv/bin/activate"

pf=("AC-030-1000.py" "AD-030-1000.py" "CD-030-0005.py" "CD-030-0010.py" "CD-030-0050.py" "CD-030-0100.py" "CD-030-1000.py")
for pfi in ${pf[@]}; do
    pfiPar=${pfi:0:2}
    python ./make_child_datacubes.py "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/$pfiPar/$pfi" "all"
done
