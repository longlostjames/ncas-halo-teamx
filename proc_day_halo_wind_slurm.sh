#!/bin/bash
#SBATCH --account=team_x
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --job-name=halo-wind
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G

# Activate conda environment
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cao_3_11

# Use provided date or default to today
if [ -z "$1" ]; then
    DATE=$(date -u +%Y%m%d)
else
    DATE=$1
fi

cd /home/users/cjwalden/git/halo-teamx

# Process for elevation 7
echo "Processing elevation 7 for $DATE"
time python proc_halo_doppy_wind.py --date "$DATE" --elevation 7

# Process for elevation 75
echo "Processing elevation 75 for $DATE"
time python proc_halo_doppy_wind.py --date "$DATE" --elevation 75

# Quicklook for elevation 7
echo "Quicklook for elevation 7 for $DATE"
time python quicklook_wind.py --date "$DATE" --elevation 7

# Quicklook for elevation 75
echo "Quicklook for elevation 75 for $DATE"
time python quicklook_wind.py --date "$DATE" --elevation 75

time python plot_ppi_map_halo.py --date "$DATE" --elevation 7

# Rsync results (adjust as needed)
rsync -avz /gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed/quicklooks/ /gws/nopw/j04/team_x/public/quicklooks/ACTA/sterzing/halo-streamline
rsync -avz /gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed/*.nc /gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed
