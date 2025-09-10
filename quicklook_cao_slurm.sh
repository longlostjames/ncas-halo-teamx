#!/bin/bash 
#SBATCH --account=ncas_obs
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --job-name=quicklook-cao
#SBATCH -o slurm_logs/quicklook_%A_%a.out
#SBATCH -e slurm_logs/quicklook_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=8G

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cao_3_11

# Convert day-of-year to actual date
target_date=$(date -d "${YEAR}-01-01 + $(($SLURM_ARRAY_TASK_ID - 1)) days" +%Y%m%d)

echo "Processing quicklook for date: $target_date"

# Run the quicklook script
python /home/users/cjwalden/git/halo-teamx/quicklook_stare_cao.py --date $target_date

echo "Completed quicklook for date: $target_date"