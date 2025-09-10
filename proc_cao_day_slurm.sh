#!/bin/bash
#SBATCH --account=team_x
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --job-name=halo-1
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G

# Activate conda environment
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cao_3_11

# Use provided date or default to today's date
if [ -z "$1" ]; then
    DATE=$(date +%Y%m%d)
else
    DATE=$1
fi


# Run the processing script with the specified date
time /home/users/cjwalden/git/halo-teamx/proc_halo_chilbolton_doppy.py --date "$DATE"
time /home/users/cjwalden/git/halo-teamx/quicklook_stare_cao.py --date "$DATE"
