#!/usr/bin/env python

import doppy
from doppy.options import OverlappedGatesOptions, OverlappedGatesMode

import os
import glob
import argparse

parser = argparse.ArgumentParser(description="Process HALO DOPPY data.")
parser.add_argument("--date", required=True, help="Date string in YYYYMMDD format")
args = parser.parse_args()

datestr = args.date

teamx_halo_path = f'/gws/pw/j07/ncas_obs_vol1/amf/raw_data/ncas-lidar-dop-1/incoming/20250603_teamx/Proc/'

yr = datestr[0:4]
mo = datestr[0:6]

datepath = os.path.join(teamx_halo_path, yr, mo, datestr)
os.chdir(datepath)

stare_files = [os.path.join(datepath, f) for f in glob.glob(f'Stare_18_{datestr}_*.hpl')]
background_files = [os.path.join(datepath, f) for f in glob.glob(f'Background_*.txt')]
print(stare_files)
print(background_files)

# Force overlapped gates with default parameters (div=2, mul=3)
opts = OverlappedGatesOptions(mode=OverlappedGatesMode.FORCE_OVERLAPPED)

stare = doppy.product.Stare.from_halo_data(
    data=stare_files,
    data_bg=background_files,
    bg_correction_method=doppy.options.BgCorrectionMethod.FIT,
    overlapped_gates=opts
)

# Construct output directory and filename
out_dir = "/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed"
os.makedirs(out_dir, exist_ok=True)
out_nc = os.path.join(out_dir, f"ncas-lidar-dop-1_{datestr}_stare_v1.0.nc")

# Write to NetCDF in the specified output directory
stare.write_to_netcdf(out_nc)