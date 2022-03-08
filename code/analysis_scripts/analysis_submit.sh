#!/bin/bash -l
#SBATCH --job-name=analysis
#SBATCH --account=pjohanss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=00:30:00
#SBATCH --partition=medium
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py386
source "/users/arawling/py386venv/bin/activate"
python ./make_child_datacubes.py "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/CD/CD-030-1000.py" "all"
