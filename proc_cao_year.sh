#!/bin/bash 
#SBATCH --account=ncas_obs
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --job-name=halo-cao-doppy
#SBATCH -o slurm_logs/%A_%a.out
#SBATCH -e slurm_logs/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cao_3_11

# Convert day-of-year to actual date
target_date=$(date -d "${YEAR}-01-01 + $(($SLURM_ARRAY_TASK_ID - 1)) days" +%Y%m%d)

time /home/users/cjwalden/git/halo-teamx/proc_halo_chilbolton_doppy.py --date $target_date