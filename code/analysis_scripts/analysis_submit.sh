#!/bin/bash -l
#SBATCH --job-name=analysis
#SBATCH --account=pjohanss
#SBATCH --time=06:00:00
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi

module purge
module restore py386
source "~/py386venv/bin/activate"
python ./make_child_datacubes.py "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/AC/AC-030-0050.py" "all"