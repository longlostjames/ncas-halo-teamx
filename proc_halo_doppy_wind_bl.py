#!/usr/bin/env python


import doppy
import os
import glob
import argparse

parser = argparse.ArgumentParser(description="Process HALO DOPPY data.")
parser.add_argument("--date", required=True, help="Date string in YYYYMMDD format")
args = parser.parse_args()

datestr = args.date

teamx_halo_path = f'/gws/pw/j07/ncas_obs_vol1/amf/raw_data/ncas-lidar-dop-1/incoming/20250603_teamx/Proc/'

yr = datestr[0:4];
mo = datestr[0:6];

datepath = os.path.join(teamx_halo_path,yr,mo,datestr);
os.chdir(datepath)

VAD_files = [os.path.join(datepath,f) for f in glob.glob(f'User5_18_{datestr}_*.hpl')];
background_files = [os.path.join(datepath,f) for f in glob.glob(f'Background_*.txt')];
print(VAD_files);
print(background_files);


# You can also pass instrument azimuth offset in degrees as an option
wind = doppy.product.Wind.from_halo_data(
    data=VAD_files,
    options=doppy.product.wind.Options(azimuth_offset_deg=286.1,overlapped_gates=True),
)

# Construct output directory and filename
out_dir = "/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed"
os.makedirs(out_dir, exist_ok=True)
out_nc = os.path.join(out_dir, f"ncas-lidar-dop-1_{datestr}_wind_v1.0.nc")

wind.write_to_netcdf(out_nc)

