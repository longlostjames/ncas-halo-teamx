#!/usr/bin/env python

import doppy
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Process HALO DOPPY data.")
parser.add_argument("--date", required=True, help="Date string in YYYYMMDD format")
args = parser.parse_args()

datestr = args.date

cao_halo_path = Path('/gws/pw/j07/ncas_obs_vol2/cao/raw_data/ncas-lidar-dop-3/data/long-term')

yr = datestr[0:4]
mo = datestr[0:6]

datepath = cao_halo_path / 'proc' / yr / mo / datestr
datepath_cross = cao_halo_path / 'proc' / 'cross' / yr / mo / datestr
background_path = cao_halo_path / 'background' / datestr

# Find stare files in datepath and cross-polarized files in datepath_cross
stare_files = list(datepath.glob(f'Stare_118_{datestr}_*.hpl'))
stare_depol_files = list(datepath_cross.glob(f'Stare_118_{datestr}_*.hpl'))
background_files = list(background_path.glob('Background_*.txt'))

print("Regular stare files:", stare_files)
print("Cross-polarized stare files:", stare_depol_files)
print("Background files:", background_files)

# Create regular stare product
stare = doppy.product.Stare.from_halo_data(
    data=stare_files,
    data_bg=background_files,
    bg_correction_method=doppy.options.BgCorrectionMethod.FIT,
    options=doppy.product.stare.Options(overlapped_gates=False),
)

# Create depolarization stare product (only if cross-polarized files exist)
if stare_depol_files:
    stare_depol = doppy.product.StareDepol.from_halo_data(
        co_data=stare_files,
        cross_data=stare_depol_files,
        data_bg=background_files,
        bg_correction_method=doppy.options.BgCorrectionMethod.FIT,
        options=doppy.product.stare.Options(overlapped_gates=False),
    )

# Construct output directory and filename with year/month structure
base_out_dir = Path("/gws/pw/j07/ncas_obs_vol2/cao/processing/ncas-lidar-dop-3/long-term/doppy_processed")
out_dir = base_out_dir / yr / mo
out_dir.mkdir(parents=True, exist_ok=True)

# Write regular stare product to NetCDF
out_nc = out_dir / f"ncas-lidar-dop-3_{datestr}_stare_v1.0.nc"
stare.write_to_netcdf(out_nc)
print(f"Created stare product: {out_nc}")

# Write depolarization stare product to NetCDF (only if files existed)
if stare_depol_files:
    out_nc_depol = out_dir / f"ncas-lidar-dop-3_{datestr}_stare-depol_v1.0.nc"
    stare_depol.write_to_netcdf(out_nc_depol)
    print(f"Created depolarization product: {out_nc_depol}")
else:
    print("No cross-polarized files found, skipping depolarization product")
