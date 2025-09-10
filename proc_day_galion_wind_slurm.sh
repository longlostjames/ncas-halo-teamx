#!/bin/bash
#SBATCH --account=team_x
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --job-name=galion-wind
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G

# Activate conda environment
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cao_3_11

# Use provided date or default to yesterday (UTC)
if [ -z "$1" ]; then
    DATE=$(date -u -d "yesterday" +%Y%m%d)
else
    DATE=$1
fi

cd /home/users/cjwalden/git/halo-teamx

# Process for elevation 7
echo "Processing Galion data for $DATE"
time /home/users/cjwalden/git/halo-teamx/proc_galion_doppy_wind.py --date "$DATE"


time python quicklook_wind_galion.py --date "$DATE" 

# Rsync results (adjust as needed)
rsync -avz /gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-wind-profiler-1/20250603_teamx/doppy_processed/quicklooks/ /gws/nopw/j04/team_x/public/quicklooks/ACTA/sterzing/galion4000

