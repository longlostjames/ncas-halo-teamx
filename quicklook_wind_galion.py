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
parser = argparse.ArgumentParser(description="Quicklook plot for Galion wind product.")
parser.add_argument("--date", required=True, help="Date string in YYYYMMDD format")
args = parser.parse_args()

datestr = args.date

# Set input directory and file path for Galion wind products
base_indir = Path("/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-wind-profiler-1/20250603_teamx/doppy_processed")
yr = datestr[0:4]
mo = datestr[0:6]
indir = base_indir / yr / mo
if not indir.exists():
    print(f"Input directory does not exist: {indir}")
    sys.exit(1)
ncfile = indir / f"ncas-lidar-wind-profiler-1_{datestr}_wind_v1.0.nc"

# Open the NetCDF file
try:
    DS = xr.open_dataset(ncfile)
except FileNotFoundError:
    print(f"File not found: {ncfile}")
    sys.exit(1)

# Convert time to datetime objects
dtime = DS['time'].values
title_date = pd.to_datetime(dtime[0]).strftime("%d-%b-%Y")
rng = DS['height'].values  # Keep in meters
hmax = 3500  # Maximum height in meters

def add_logos(fig, plot_center):
    """Add AMOF and NCAS logos to figure"""
    # Add AMOF logo as an inset axis in the bottom left outside the plot
    logo_path = "/home/users/cjwalden/git/halo-teamx/amof-web-header-wr.png"
    logo_img = mpimg.imread(logo_path)
    logo_h = 0.04
    logo_w = logo_h * (logo_img.shape[1] / logo_img.shape[0])
    logo_ax = fig.add_axes([0.01, 0.01, logo_w, logo_h])
    logo_ax.imshow(logo_img)
    logo_ax.axis('off')

    # Add NCAS logo as an inset axis in the top right
    ncas_logo_path = "/home/users/cjwalden/git/halo-teamx/NCAS_national_centre_logo_transparent.png"
    ncas_logo_img = mpimg.imread(ncas_logo_path)
    ncas_logo_h = 0.05
    ncas_logo_w = ncas_logo_h * (ncas_logo_img.shape[1] / ncas_logo_img.shape[0])
    ncas_logo_ax = fig.add_axes([1.0 - ncas_logo_w - 0.01, 0.94, ncas_logo_w, ncas_logo_h])
    ncas_logo_ax.imshow(ncas_logo_img)
    ncas_logo_ax.axis('off')

# =============================================================================
# FIGURE 1: Wind Components (U, V, W)
# =============================================================================
fig1 = plt.figure(figsize=(14, 12))
fig1.subplots_adjust(hspace=0.4, top=0.88, bottom=0.10)

ax1 = fig1.add_subplot(311)
ax2 = fig1.add_subplot(312)
ax3 = fig1.add_subplot(313)

ax1.set_ylim(0, hmax)
ax2.set_ylim(0, hmax)
ax3.set_ylim(0, hmax)

myFmt = mdates.DateFormatter('%H:%M')
ax1.xaxis.set_major_formatter(myFmt)
ax2.xaxis.set_major_formatter(myFmt)
ax3.xaxis.set_major_formatter(myFmt)

# Plot u-wind component
if 'uwind' in DS.variables:
    h1 = ax1.pcolormesh(
        dtime[:],
        rng,
        DS['uwind'][:, :].transpose(),
        vmin=-15, vmax=15,
        cmap=cmocean.cm.balance,
        shading='auto'
    )
    cb1 = plt.colorbar(h1, ax=ax1, orientation='vertical')
    cb1.set_label('U-wind component (m s$^{-1}$)')
else:
    ax1.text(0.5, 0.5, 'uwind not found', ha='center', va='center', transform=ax1.transAxes)
ax1.set_ylabel('Height (m AGL)')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax1.set_xlim(dtime[0], dtime[-1])

# Plot v-wind component
if 'vwind' in DS.variables:
    h2 = ax2.pcolormesh(
        dtime[:],
        rng,
        DS['vwind'][:, :].transpose(),
        vmin=-15, vmax=15,
        cmap=cmocean.cm.balance,
        shading='auto'
    )
    cb2 = plt.colorbar(h2, ax=ax2, orientation='vertical')
    cb2.set_label('V-wind component (m s$^{-1}$)')
else:
    ax2.text(0.5, 0.5, 'vwind not found', ha='center', va='center', transform=ax2.transAxes)
ax2.set_ylabel('Height (m AGL)')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.set_xlim(dtime[0], dtime[-1])

# Plot w-wind component (vertical velocity)
if 'wwind' in DS.variables:
    h3 = ax3.pcolormesh(
        dtime[:],
        rng,
        DS['wwind'][:, :].transpose(),
        vmin=-3, vmax=3,  # Smaller range for vertical velocity
        cmap=cmocean.cm.balance,
        shading='auto'
    )
    cb3 = plt.colorbar(h3, ax=ax3, orientation='vertical')
    cb3.set_label('W-wind component (m s$^{-1}$)')
else:
    ax3.text(0.5, 0.5, 'wwind not found', ha='center', va='center', transform=ax3.transAxes)
ax3.set_ylabel('Height (m AGL)')
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax3.set_xlabel('Time (UTC)')
ax3.set_xlim(dtime[0], dtime[-1])

plt.tight_layout(rect=[0, 0.04, 1, 0.93])

# Calculate plot center for title placement
axes1 = [ax1, ax2, ax3]
lefts1 = [ax.get_position().x0 for ax in axes1]
rights1 = [ax.get_position().x1 for ax in axes1]
plot_left1 = min(lefts1)
plot_right1 = max(rights1)
plot_center1 = plot_left1 + (plot_right1 - plot_left1) / 2

# Add logos and title to figure 1
add_logos(fig1, plot_center1)
fig1.suptitle(
    f"NCAS Wind Profiler 1, TEAMx Sterzing Wind Components {title_date}",
    fontsize=16,
    y=0.96,
    x=plot_center1
)

# =============================================================================
# FIGURE 2: Wind Speed and Direction
# =============================================================================
fig2 = plt.figure(figsize=(14, 8))
fig2.subplots_adjust(hspace=0.4, top=0.88, bottom=0.10)

ax4 = fig2.add_subplot(211)
ax5 = fig2.add_subplot(212)

ax4.set_ylim(0, hmax)
ax5.set_ylim(0, hmax)

ax4.xaxis.set_major_formatter(myFmt)
ax5.xaxis.set_major_formatter(myFmt)

# Plot wind speed (calculated from u and v components)
if 'uwind' in DS.variables and 'vwind' in DS.variables:
    wind_speed = np.sqrt(DS['uwind']**2 + DS['vwind']**2)
    h4 = ax4.pcolormesh(
        dtime[:],
        rng,
        wind_speed[:, :].transpose(),
        vmin=0, vmax=15,
        cmap='viridis',
        shading='auto'
    )
    cb4 = plt.colorbar(h4, ax=ax4, orientation='vertical')
    cb4.set_label('Wind speed (m s$^{-1}$)')
else:
    ax4.text(0.5, 0.5, 'Wind components not found', ha='center', va='center', transform=ax4.transAxes)
ax4.set_ylabel('Height (m AGL)')
ax4.grid(True, which='both', linestyle='--', linewidth=0.5)
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax4.set_xlim(dtime[0], dtime[-1])

# Plot wind direction (calculated from u and v components)
if 'uwind' in DS.variables and 'vwind' in DS.variables:
    # Calculate wind direction (meteorological convention)
    wind_dir = (270 - np.arctan2(DS['vwind'], DS['uwind']) * 180/np.pi) % 360
    h5 = ax5.pcolormesh(
        dtime[:],
        rng,
        wind_dir[:, :].transpose(),
        vmin=0, vmax=360,
        cmap='twilight_shifted',
        shading='auto'
    )
    cb5 = plt.colorbar(h5, ax=ax5, orientation='vertical')
    cb5.set_label('Wind direction (degrees)')
else:
    ax5.text(0.5, 0.5, 'Wind components not found', ha='center', va='center', transform=ax5.transAxes)
ax5.set_ylabel('Height (m AGL)')
ax5.grid(True, which='both', linestyle='--', linewidth=0.5)
ax5.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax5.set_xlabel('Time (UTC)')
ax5.set_xlim(dtime[0], dtime[-1])

plt.tight_layout(rect=[0, 0.04, 1, 0.93])

# Calculate plot center for title placement
axes2 = [ax4, ax5]
lefts2 = [ax.get_position().x0 for ax in axes2]
rights2 = [ax.get_position().x1 for ax in axes2]
plot_left2 = min(lefts2)
plot_right2 = max(rights2)
plot_center2 = plot_left2 + (plot_right2 - plot_left2) / 2

# Add logos and title to figure 2
add_logos(fig2, plot_center2)
fig2.suptitle(
    f"NCAS Wind Profiler 1, TEAMx Sterzing Wind Speed & Direction {title_date}",
    fontsize=16,
    y=0.96,
    x=plot_center2
)

# Save both figures
quicklook_dir = base_indir / "quicklooks" 
quicklook_dir.mkdir(parents=True, exist_ok=True)

uvw_dir = quicklook_dir / "uvw"
sd_dir = quicklook_dir / "speed-direction"

uvw_dir.mkdir(parents=True, exist_ok=True)
sd_dir.mkdir(parents=True, exist_ok=True)

out_png1 = uvw_dir / f'ncas-lidar-wind-profiler-1_sterzing_galion_wind_components_{datestr}.png'
out_png2 = sd_dir / f'ncas-lidar-wind-profiler-1_sterzing_galion_wind_speed_dir_{datestr}.png'

fig1.savefig(out_png1, dpi=150)
fig2.savefig(out_png2, dpi=150)

plt.show()
