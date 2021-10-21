#!/bin/bash -l
#SBATCH --job-name=inertia
#SBATCH --account=pjohanss
#SBATCH --time=6:00:00
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi
#SBATCH --array=0

module purge
module restore py386

case $SLURM_ARRAY_TASK_ID in
	0) oc=("NGCa0524t" "NGCa3348t") ;;
	1) oc=() ;;
esac

for i in ${oc[@]}
do
    python ./inertia_analysis.py "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/triaxial/"$i"/output" -v -n -i -f stars 
done
