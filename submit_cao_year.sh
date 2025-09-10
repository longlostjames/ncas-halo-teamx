#!/bin/bash
# submit_year.sh

YEAR=${1:-2016}

# Calculate days in year using date arithmetic
DAYS=$(date -d "$((YEAR + 1))-01-01 - 1 day" +%j)

echo "Processing year $YEAR with $DAYS days"

# Submit with exact array size
sbatch --array=1-$DAYS --export=YEAR=$YEAR proc_cao_year.sh