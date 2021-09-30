#!/bin/bash -l
#SBATCH --job-name=inertia
#SBATCH --account=pjohanss
#SBATCH --time=24:00:00
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi
#SBATCH --array=0

module purge
module restore py386

case $SLURM_ARRAY_TASK_ID in
	0) oc=("0-001" "0-005" "0-030" "0-180" "1-000") ;;
	1) oc=() ;;
esac

for i in ${oc[@]}
do
    python ./inertia_analysis.py "/scratch/pjohanss/arawling/collisionless_merger/res-tests/fiducial/"$i"/output" -v -n
done
