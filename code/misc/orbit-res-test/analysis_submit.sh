#!/bin/bash -l
#SBATCH --job-name=inertia
#SBATCH --account=pjohanss
#SBATCH --time=24:00:00
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi
#SBATCH --array=0-1

module purge
module restore py386

case $SLURM_ARRAY_TASK_ID in
	0) oc=("0-001" "0-005") ;;
	1) oc=("0-030" "0-180" "1-000") ;;
esac

#oc=("0-005" "0-030" "0-180" "1-000")
for i in ${oc[@]}
do
    python ./inertia_2.py "/scratch/pjohanss/arawling/collisionless_merger/res-tests/x10/"$i"/output" -v -n
done
