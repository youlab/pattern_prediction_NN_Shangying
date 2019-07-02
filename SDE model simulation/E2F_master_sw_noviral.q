#!/bin/bash
#
#SBATCH --array=1-1000
#SBATCH -o slurm.out
#SBATCH --mem=2G
#SBATCH -e slurm.err

/opt/apps/matlabR2016a/bin/matlab -nojvm -nodisplay -singleCompThread -r "rank=$SLURM_ARRAY_TASK_ID;E2F_master_sw_noviral;quit"
