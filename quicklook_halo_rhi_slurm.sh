#!/bin/bash
#SBATCH --account=team_x
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --job-name=quicklook-halo-rhi
#SBATCH -o slurm_logs/halo_rhi_%A_%a.out
#SBATCH -e slurm_logs/halo_rhi_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cao_3_11

# Convert day-of-year to actual date
target_date=$(date -d "${YEAR}-01-01 +$((SLURM_ARRAY_TASK_ID - 1)) days" +%Y%m%d)

echo "Processing HALO RHI quicklook for date: $target_date"
echo "Year: $YEAR, Array task ID: $SLURM_ARRAY_TASK_ID"

# Run the quicklook script
python /home/users/cjwalden/git/halo-teamx/quicklook_rhi_halo.py --date $target_date

echo "Completed HALO RHI quicklook for date: $target_date"