#!/bin/bash

# Submit script for HALO PPI quicklooks
# Usage: ./submit_halo_ppi.sh START_DATE [END_DATE]
# Example: ./submit_halo_ppi.sh 20250626 20250630

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 START_DATE [END_DATE]"
    echo "Example: $0 20250626 20250630"
    echo "Dates must be in YYYYMMDD format"
    echo "If END_DATE not provided, will process only START_DATE"
    exit 1
fi

START_DATE=$1
END_DATE=${2:-$START_DATE}

# Validate date format
if [[ ! $START_DATE =~ ^[0-9]{8}$ ]] || [[ ! $END_DATE =~ ^[0-9]{8}$ ]]; then
    echo "Error: Dates must be in YYYYMMDD format"
    echo "Usage: $0 START_DATE [END_DATE]"
    echo "Example: $0 20250626 20250630"
    exit 1
fi

# Extract year and validate both dates are in the same year
START_YEAR=${START_DATE:0:4}
END_YEAR=${END_DATE:0:4}

if [ "$START_YEAR" != "$END_YEAR" ]; then
    echo "Error: START_DATE and END_DATE must be in the same year"
    echo "START_DATE year: $START_YEAR"
    echo "END_DATE year: $END_YEAR"
    exit 1
fi

YEAR=$START_YEAR

# Check if dates are valid
if ! date -d "${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}" >/dev/null 2>&1; then
    echo "Error: Invalid START_DATE: $START_DATE"
    exit 1
fi

if ! date -d "${END_DATE:0:4}-${END_DATE:4:2}-${END_DATE:6:2}" >/dev/null 2>&1; then
    echo "Error: Invalid END_DATE: $END_DATE"
    exit 1
fi

# Check if start date is before or equal to end date
if [ $START_DATE -gt $END_DATE ]; then
    echo "Error: START_DATE ($START_DATE) must be before or equal to END_DATE ($END_DATE)"
    exit 1
fi

# Convert YYYYMMDD to day-of-year
jan1="${YEAR}-01-01"
start_formatted="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"
end_formatted="${END_DATE:0:4}-${END_DATE:4:2}-${END_DATE:6:2}"

START_DOY=$(( ($(date -d "$start_formatted" +%s) - $(date -d "$jan1" +%s)) / 86400 + 1 ))
END_DOY=$(( ($(date -d "$end_formatted" +%s) - $(date -d "$jan1" +%s)) / 86400 + 1 ))

echo "Submitting HALO PPI quicklook jobs"
echo "Year: $YEAR"
echo "Date range: $start_formatted to $end_formatted"
echo "Day-of-year range: $START_DOY to $END_DOY"

# Create logs directory
mkdir -p slurm_logs

# Calculate number of days
num_days=$((END_DOY - START_DOY + 1))

# Submit the job array with YEAR exported
job_id=$(sbatch --array=${START_DOY}-${END_DOY} \
    --export=YEAR=${YEAR} \
    quicklook_halo_ppi_slurm.sh | awk '{print $4}')

if [ $? -eq 0 ]; then
    echo "Job submitted successfully with ID: $job_id"
    echo "Processing $num_days day(s)"
    echo "Monitor progress with: squeue -u $USER"
    echo "Check logs in: slurm_logs/halo_ppi_${job_id}_*.out"
    echo ""
    echo "Output will be saved to:"
    echo "/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed/quicklooks/"
else
    echo "Error submitting job"
    exit 1
fi