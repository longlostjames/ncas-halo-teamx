#!/bin/bash
# submit_year.sh - Submit quicklook generation for a whole year

YEAR=${1:-2016}

# Calculate days in year using date arithmetic
DAYS=$(date -d "$((YEAR + 1))-01-01 - 1 day" +%j)

echo "Submitting quicklook generation for year $YEAR with $DAYS days"

# Submit with exact array size
sbatch --array=1-$DAYS --export=YEAR=$YEAR quicklook_cao_slurm.sh