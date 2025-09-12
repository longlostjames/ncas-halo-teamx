#!/bin/bash
#SBATCH --account=team_x
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --job-name=quicklook-halo-ppi
#SBATCH -o slurm_logs/halo_ppi_%A_%a.out
#SBATCH -e slurm_logs/halo_ppi_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate halo-teamx

# Convert day-of-year to actual date (fixed syntax)
target_date=$(date -d "${YEAR}-01-01 +$((SLURM_ARRAY_TASK_ID - 1)) days" +%Y%m%d)

echo "Processing HALO PPI quicklook for date: $target_date"
echo "Year: $YEAR, Array task ID: $SLURM_ARRAY_TASK_ID"

# Run the quicklook script
python /home/users/cjwalden/git/halo-teamx/quicklook_ppi_halo.py --date $target_date

echo "Completed HALO PPI quicklook for date: $target_date"