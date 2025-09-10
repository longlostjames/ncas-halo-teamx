#!/usr/bin/env python

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import matplotlib.dates as mdates
import argparse
import sys
import os
import cftime
import cmocean
import pandas as pd
import pyart
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path


# Parse date from command line
parser = argparse.ArgumentParser(description="Quicklook plot for HALO stare product.")
parser.add_argument("--date", required=True, help="Date string in YYYYMMDD format")
args = parser.parse_args()

datestr = args.date

# Set input directory and file path
base_indir = Path("/gws/pw/j07/ncas_obs_vol2/cao/processing/ncas-lidar-dop-3/long-term/doppy_processed")
indir = base_indir / datestr[0:4] / datestr[0:6]
if not indir.exists():
    print(f"Input directory does not exist: {indir}")
    sys.exit(1)
ncfile = indir / f"ncas-lidar-dop-3_{datestr}_stare_v1.0.nc"

# Open the NetCDF file
try:
    DS = xr.open_dataset(ncfile)
except FileNotFoundError:
    print(f"File not found: {ncfile}")
    sys.exit(1)

# Convert time to datetime objects
dtime = DS['time'].values

fig = plt.figure(figsize=(14, 14))  # Increase figure height for 4 panels
fig.subplots_adjust(hspace=0.5, top=0.88, bottom=0.10)  # Make more space at the top for the logo

ax1 = fig.add_subplot(411)  # Change to 4 rows
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)  # Add fourth panel

hmax = 10

ax1.set_ylim(0, hmax) 
ax2.set_ylim(0, hmax)
ax3.set_ylim(0, hmax)
ax4.set_ylim(0, hmax)  # Set y-limit for SNR panel

title_date = pd.to_datetime(dtime[0]).strftime("%d-%b-%Y")

myFmt = mdates.DateFormatter('%H:%M')
ax1.xaxis.set_major_formatter(myFmt)
ax2.xaxis.set_major_formatter(myFmt)
ax3.xaxis.set_major_formatter(myFmt)
ax4.xaxis.set_major_formatter(myFmt)  # Add formatter for SNR panel

rng = DS['range'].values / 1000.0  # Convert range to km

# Plot attenuated backscatter (log scale)
beta = DS['beta'].values
h1 = ax1.pcolormesh(
    dtime[:],
    rng,
    beta.transpose(),
    norm=colors.LogNorm(vmin=1e-8, vmax=1e-4),
    cmap = 'viridis',
    shading='auto'
)
cb1 = plt.colorbar(h1, ax=ax1, orientation='vertical')
cb1.set_label('Attenuated backscatter (m$^{-1}$ sr$^{-1}$)')
ax1.set_ylabel('Range (km)')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax1.set_xlim(dtime[0], dtime[-1])

# Plot raw attenuated backscatter (log scale)
if 'beta_raw' in DS.variables:
    beta_raw = DS['beta_raw'].values
    h2 = ax2.pcolormesh(
        dtime[:],
        rng,
        beta_raw.transpose(),
        norm=colors.LogNorm(vmin=1e-8, vmax=1e-4),
        cmap='viridis',
        shading='auto'
    )
    cb2 = plt.colorbar(h2, ax=ax2, orientation='vertical')
    cb2.set_label('Raw attenuated backscatter (m$^{-1}$ sr$^{-1}$)')
else:
    ax2.text(0.5, 0.5, 'beta_raw not found', ha='center', va='center', transform=ax2.transAxes)
ax2.set_ylabel('Range (km)')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.set_xlim(dtime[0], dtime[-1])

# Plot Doppler velocity
h3 = ax3.pcolormesh(
    dtime[:],
    rng,
    DS['v'][:, :].transpose(),
    vmin=-10, vmax=10,
    cmap=cmocean.cm.balance,
    shading='auto'
)
cb3 = plt.colorbar(h3, ax=ax3, orientation='vertical')
cb3.set_label('Doppler velocity (m s$^{-1}$)')
ax3.set_ylabel('Range (km)')
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax3.set_xlim(dtime[0], dtime[-1])

# Plot SNR
if 'snr' in DS.variables:
    snr = DS['snr'].values
    # Convert SNR to dB: SNR_dB = 10 * log10(SNR)
    snr_db = 10 * np.log10(snr)
    h4 = ax4.pcolormesh(
        dtime[:],
        rng,
        snr_db.transpose(),
        vmin=-20, vmax=20,  # Adjust range as needed for dB values
        cmap='viridis',
        shading='auto'
    )
    cb4 = plt.colorbar(h4, ax=ax4, orientation='vertical')
    cb4.set_label('SNR (dB)')
else:
    ax4.text(0.5, 0.5, 'SNR not found', ha='center', va='center', transform=ax4.transAxes)
ax4.set_ylabel('Range (km)')
ax4.grid(True, which='both', linestyle='--', linewidth=0.5)
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax4.set_xlabel('Time (UTC)')  # Move x-label to bottom panel
ax4.set_xlim(dtime[0], dtime[-1])

plt.tight_layout(rect=[0, 0.04, 1, 0.93])  # Leave more space at the top and bottom for logos

# Calculate the center of the plot region (not including colorbars)
axes = [ax1, ax2, ax3, ax4]  # Include ax4 in the list
lefts = [ax.get_position().x0 for ax in axes]
rights = [ax.get_position().x1 for ax in axes]
plot_left = min(lefts)
plot_right = max(rights)
plot_center = plot_left + (plot_right - plot_left) / 2

# Add AMOF logo as an inset axis in the bottom left outside the plot
logo_path = "/home/users/cjwalden/git/halo-teamx/amof-web-header-wr.png"
logo_img = mpimg.imread(logo_path)
logo_h = 0.04
logo_w = logo_h * (logo_img.shape[1] / logo_img.shape[0])
logo_ax = fig.add_axes([0.01, 0.01, logo_w, logo_h])
logo_ax.imshow(logo_img)
logo_ax.axis('off')

# Add NCAS logo as an inset axis in the top right, move up very slightly
ncas_logo_path = "/home/users/cjwalden/git/halo-teamx/NCAS_national_centre_logo_transparent.png"
ncas_logo_img = mpimg.imread(ncas_logo_path)
ncas_logo_h = 0.05
ncas_logo_w = ncas_logo_h * (ncas_logo_img.shape[1] / ncas_logo_img.shape[0])
ncas_logo_ax = fig.add_axes([1.0 - ncas_logo_w - 0.01, 0.94, ncas_logo_w, ncas_logo_h])  # was 0.925, now 0.94
ncas_logo_ax.imshow(ncas_logo_img)
ncas_logo_ax.axis('off')

# Use suptitle for the main figure title, centered over the plot region
fig.suptitle(
    f"NCAS Doppler Lidar 3, Chilbolton {title_date}",
    fontsize=16,
    y=0.96,
    x=plot_center  # Center over the plot region
)

# Create quicklooks output directory if it doesn't exist
quicklook_dir = base_indir / "quicklooks" / datestr[0:4]
quicklook_dir.mkdir(parents=True, exist_ok=True)
out_png = quicklook_dir / f'ncas-lidar-dop-3_cao_stare_{datestr}_v1.0.png'
plt.savefig(out_png, dpi=150)
plt.show()