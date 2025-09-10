#!/bin/bash 
#SBATCH --account=team_x
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --job-name=halo-1
#SBATCH -o slurm_logs/%A_%a.out
#SBATCH -e slurm_logs/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --array=1-9
#SBATCH --mem=64G

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cao_3_11

time /home/users/cjwalden/git/halo-teamx/proc_halo_doppy.py --date 2025070${SLURM_ARRAY_TASK_ID}
