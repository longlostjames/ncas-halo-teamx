#!/usr/bin/env python

import doppy
from pathlib import Path
import argparse
from doppy.raw import GalionScn
from doppy.product import Wind

parser = argparse.ArgumentParser(description="Process Galion DOPPY wind data.")
parser.add_argument("--date", required=True, help="Date string in YYYYMMDD format")
args = parser.parse_args()

datestr = args.date

# Set base path for Galion data
galion_base_path = Path('/gws/pw/j07/ncas_obs_vol1/amf/raw_data/ncas-lidar-wind-profiler-1/incoming/20250603_teamx')

yr = datestr[0:4]
mo = datestr[0:6]
day = datestr

# Construct path to Galion scan files (files are in hourly subdirectories 00, 01, etc.)
galion_path = galion_base_path / yr / mo / day

# Find all .scn files for the given date (searches recursively in hour subdirectories)
galion_files = list(galion_path.rglob("*.scn"))

print(f"Found {len(galion_files)} Galion scan files for {datestr}")
print(f"Searching in: {galion_path}")
if galion_files:
    print("Sample files:")
    for file in galion_files[:5]:  # Print first 5 files with their hour directories
        print(f"  {file}")

if not galion_files:
    print(f"No Galion files found for date {datestr}")
    # List available hour directories for debugging
    hour_dirs = [d for d in galion_path.iterdir() if d.is_dir()]
    print(f"Available hour directories: {hour_dirs}")
    exit(1)

# Generate wind product (removed Wind.Options, pass azimuth_offset_deg directly)
wind = Wind.from_galion_data(galion_files, azimuth_offset_deg=70.1)  # Adjust offset as needed

# Construct output directory and filename
base_out_dir = Path("/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-wind-profiler-1/20250603_teamx/doppy_processed")
out_dir = base_out_dir / yr / mo
out_dir.mkdir(parents=True, exist_ok=True)

# Save results
out_nc = out_dir / f"ncas-lidar-wind-profiler-1_{datestr}_wind_v1.0.nc"
wind.write_to_netcdf(out_nc)

print(f"Created wind product: {out_nc}")