#!/bin/bash 
#SBATCH --account=team_x
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --job-name=halo-1
#SBATCH -o slurm_logs/%A_%a.out
#SBATCH -e slurm_logs/%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cao_3_11

# Use provided date or default to yesterday's date
if [ -z "$1" ]; then
    DATE=$(date -d "yesterday" +%Y%m%d)
else
    DATE=$1
fi


time /home/users/cjwalden/git/halo-teamx/quicklook_wind.py --date "$DATE"
#rsync -avz /gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed/quicklooks/ /gws/nopw/j04/team_x/public/quicklooks/ACTA/sterzing/lidar/
