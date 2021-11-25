#!/bin/bash -l
#SBATCH --job-name=analysis
#SBATCH --account=pjohanss
#SBATCH --time=6:00:00
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexander.rawlings@helsinki.fi
#SBATCH --array=0,1

module purge
module restore py386

#python ./inertia_analysis.py "/scratch/pjohanss/arawling/collisionless_merger/regen-test/original/output" -v -n -f stars

case $SLURM_ARRAY_TASK_ID in
	0) oc=("NGCa0524" "NGCa3348" "NGCa3607") ;;
	1) oc=("NGCa4291" "NGCa2986") ;;
esac

for i in ${oc[@]}
do
    python ./bh_perturb_distribution.py "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/starsoft10pc/$i/output" -n 
done