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
import glob
import datetime

# Parse date from command line
parser = argparse.ArgumentParser(description="Quicklook plot for HALO wind product.")
parser.add_argument("--date", required=True, help="Date string in YYYYMMDD format")
parser.add_argument("--elevation", type=float, default=None, help="Elevation angle (optional, will auto-detect if not specified)")
args = parser.parse_args()

datestr = args.date
elevation_filter = args.elevation

# Set input directory and file path
indir = Path("/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed")

# Look for wind files - try different naming patterns
if elevation_filter is not None:
    # User specified elevation
    ncfile = indir / f"ncas-lidar-dop-1_{datestr}_wind_elev{elevation_filter:g}_v1.0.nc"
    if not ncfile.exists():
        print(f"File not found: {ncfile}")
        sys.exit(1)
else:
    # Auto-detect wind files
    pattern1 = str(indir / f"ncas-lidar-dop-1_{datestr}_wind_elev*_v1.0.nc")
    pattern2 = str(indir / f"ncas-lidar-dop-1_{datestr}_wind_v1.0.nc")
    
    wind_files = glob.glob(pattern1) + glob.glob(pattern2)
    
    if not wind_files:
        print(f"No wind files found for date {datestr} in {indir}")
        print("Looking for files matching:")
        print(f"  {pattern1}")
        print(f"  {pattern2}")
        sys.exit(1)
    elif len(wind_files) == 1:
        ncfile = Path(wind_files[0])
        print(f"Found wind file: {ncfile.name}")
    else:
        print(f"Multiple wind files found:")
        for f in wind_files:
            print(f"  {Path(f).name}")
        print("Please specify elevation with --elevation parameter")
        
        # Extract elevations from filenames
        elevations = []
        for f in wind_files:
            filename = Path(f).name
            if 'elev' in filename:
                try:
                    elev_part = filename.split('elev')[1].split('_')[0]
                    elevation = float(elev_part)
                    elevations.append(elevation)
                except:
                    pass
        
        if elevations:
            print(f"Available elevations: {sorted(set(elevations))}")
            print(f"Example: python {sys.argv[0]} --date {datestr} --elevation {sorted(set(elevations))[0]}")
        
        sys.exit(1)

print(f"Using file: {ncfile}")

# Open the NetCDF file
try:
    DS = xr.open_dataset(ncfile)
    print(f"Successfully opened: {ncfile}")
    print(f"Variables in file: {list(DS.variables.keys())}")
except FileNotFoundError:
    print(f"File not found: {ncfile}")
    sys.exit(1)
except Exception as e:
    print(f"Error opening file {ncfile}: {e}")
    sys.exit(1)

# Convert time to datetime objects
dtime = DS['time'].values

fig = plt.figure(figsize=(14, 16))  # Increase height for 4 panels
fig.subplots_adjust(hspace=0.5, top=0.88, bottom=0.10)

ax1 = fig.add_subplot(411)  # Change to 4 rows
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

# Extract elevation from filename for calculating appropriate height limit
elevation_angle = None
filename = ncfile.name
if 'elev' in filename:
    try:
        elev_part = filename.split('elev')[1].split('_')[0]
        elevation_angle = float(elev_part)
    except:
        elevation_angle = None

# Calculate appropriate height limit based on elevation angle
if elevation_angle is not None and elevation_angle < 10:
    # For low elevation angles, calculate max height at 4 km range
    # height = range * sin(elevation_angle)
    max_range_km = 4.0  # 4 km maximum range
    elevation_rad = np.radians(elevation_angle)
    hmax = max_range_km * np.sin(elevation_rad)
    # Add some buffer (20%) and round up to nearest 0.1 km
    hmax = np.ceil(hmax * 1.2 * 10) / 10
    print(f"Using height limit of {hmax:.1f} km for {elevation_angle:.0f}° elevation")
else:
    # For higher elevations or unknown elevation, use default 10 km
    hmax = 10
    print(f"Using default height limit of {hmax} km")

ax1.set_ylim(0, hmax)
ax2.set_ylim(0, hmax)
ax3.set_ylim(0, hmax)
ax4.set_ylim(0, hmax)

title_date = pd.to_datetime(dtime[0]).strftime("%d-%b-%Y")

myFmt = mdates.DateFormatter('%H:%M')
ax1.xaxis.set_major_formatter(myFmt)
ax2.xaxis.set_major_formatter(myFmt)
ax3.xaxis.set_major_formatter(myFmt)
ax4.xaxis.set_major_formatter(myFmt)

rng = DS['height'].values / 1000  # Convert from meters to kilometers

# Plot u-wind component
if 'uwind' in DS.variables:
    h1 = ax1.pcolormesh(
        dtime[:],
        rng,
        DS['uwind'][:, :].transpose(),
        vmin=-20, vmax=20,
        cmap=cmocean.cm.balance,
        shading='auto'
    )
    cb1 = plt.colorbar(h1, ax=ax1, orientation='vertical')
    cb1.set_label('U-wind component (m s$^{-1}$)')
else:
    ax1.text(0.5, 0.5, 'uwind not found', ha='center', va='center', transform=ax1.transAxes)
ax1.set_ylabel('Height (km AGL)')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax1.set_xlim(dtime[0], dtime[-1])

# Plot v-wind component
if 'vwind' in DS.variables:
    h2 = ax2.pcolormesh(
        dtime[:],
        rng,
        DS['vwind'][:, :].transpose(),
        vmin=-20, vmax=20,
        cmap=cmocean.cm.balance,
        shading='auto'
    )
    cb2 = plt.colorbar(h2, ax=ax2, orientation='vertical')
    cb2.set_label('V-wind component (m s$^{-1}$)')
else:
    ax2.text(0.5, 0.5, 'vwind not found', ha='center', va='center', transform=ax2.transAxes)
ax2.set_ylabel('Height (km AGL)')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.set_xlim(dtime[0], dtime[-1])

# Plot raw u-wind component
if 'uwind_raw' in DS.variables:
    h3 = ax3.pcolormesh(
        dtime[:],
        rng,
        DS['uwind_raw'][:, :].transpose(),
        vmin=-20, vmax=20,
        cmap=cmocean.cm.balance,
        shading='auto'
    )
    cb3 = plt.colorbar(h3, ax=ax3, orientation='vertical')
    cb3.set_label('Raw U-wind component (m s$^{-1}$)')
else:
    ax3.text(0.5, 0.5, 'uwind_raw not found', ha='center', va='center', transform=ax3.transAxes)
ax3.set_ylabel('Height (km AGL)')
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax3.set_xlim(dtime[0], dtime[-1])

# Plot raw v-wind component
if 'vwind_raw' in DS.variables:
    h4 = ax4.pcolormesh(
        dtime[:],
        rng,
        DS['vwind_raw'][:, :].transpose(),
        vmin=-20, vmax=20,
        cmap=cmocean.cm.balance,
        shading='auto'
    )
    cb4 = plt.colorbar(h4, ax=ax4, orientation='vertical')
    cb4.set_label('Raw V-wind component (m s$^{-1}$)')
else:
    ax4.text(0.5, 0.5, 'vwind_raw not found', ha='center', va='center', transform=ax4.transAxes)
ax4.set_ylabel('Height (km AGL)')  # Fixed: was showing 'km' instead of 'm'
ax4.grid(True, which='both', linestyle='--', linewidth=0.5)
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax4.set_xlabel('Time (UTC)')  # Move x-label to bottom panel
ax4.set_xlim(dtime[0], dtime[-1])

plt.tight_layout(rect=[0, 0.04, 1, 0.93])

# Calculate the center of the plot region (not including colorbars)
axes = [ax1, ax2, ax3, ax4]
lefts = [ax.get_position().x0 for ax in axes]
rights = [ax.get_position().x1 for ax in axes]
plot_left = min(lefts)
plot_right = max(rights)
plot_center = plot_left + (plot_right - plot_left) / 2

# Add AMOF logo as an inset axis in the bottom left outside the plot
logo_path = "/home/users/cjwalden/git/halo-teamx/amof-web-header-wr.png"
if os.path.exists(logo_path):
    logo_img = mpimg.imread(logo_path)
    logo_h = 0.04
    logo_w = logo_h * (logo_img.shape[1] / logo_img.shape[0])
    logo_ax = fig.add_axes([0.01, 0.01, logo_w, logo_h])
    logo_ax.imshow(logo_img)
    logo_ax.axis('off')

# Add NCAS logo as an inset axis in the top right
ncas_logo_path = "/home/users/cjwalden/git/halo-teamx/NCAS_national_centre_logo_transparent.png"
if os.path.exists(ncas_logo_path):
    ncas_logo_img = mpimg.imread(ncas_logo_path)
    ncas_logo_h = 0.05
    ncas_logo_w = ncas_logo_h * (ncas_logo_img.shape[1] / ncas_logo_img.shape[0])
    ncas_logo_ax = fig.add_axes([1.0 - ncas_logo_w - 0.01, 0.94, ncas_logo_w, ncas_logo_h])
    ncas_logo_ax.imshow(ncas_logo_img)
    ncas_logo_ax.axis('off')

# Extract elevation from filename for title
elevation_str = ""
filename = ncfile.name
if 'elev' in filename:
    try:
        elev_part = filename.split('elev')[1].split('_')[0]
        elevation = float(elev_part)
        elevation_str = f" (Elevation: {elevation:.0f}°)"
    except:
        pass

# Use suptitle for the main figure title
fig.suptitle(
    f"NCAS Doppler Lidar 1, TEAMx Sterzing Wind Components {title_date}{elevation_str}",
    fontsize=16,
    y=0.96,
    x=plot_center
)


# Calculate start and end of day for the given date
plot_start = pd.to_datetime(datestr, format="%Y%m%d")
plot_end = plot_start + pd.Timedelta(days=1)

# After creating each axis (ax1, ax2, ax3, ax4, ax_speed, ax_dir, ax_speed_raw), set:
ax1.set_xlim(plot_start, plot_end)
ax2.set_xlim(plot_start, plot_end)
ax3.set_xlim(plot_start, plot_end)
ax4.set_xlim(plot_start, plot_end)

# Create quicklooks output directory if it doesn't exist
quicklook_dir = indir / "quicklooks"

# Determine subdirectory based on elevation angle
if elevation_angle is not None and elevation_angle < 10:
    # For low elevations (boundary layer), use wind_bl subdirectory
    plot_subdir = quicklook_dir / "wind_bl"
    print(f"Using boundary layer subdirectory for {elevation_angle:.0f}° elevation")
else:
    # For higher elevations or unknown, use wind subdirectory
    plot_subdir = quicklook_dir / "wind"
    print(f"Using standard wind subdirectory")

plot_subdir.mkdir(parents=True, exist_ok=True)

# Generate output filename with elevation info if available
if elevation_str:
    elev_suffix = filename.split('elev')[1].split('_')[0]
    out_png = plot_subdir / f'ncas-lidar-dop-1_sterzing_wind_elev{elev_suffix}_{datestr}.png'
else:
    out_png = plot_subdir / f'ncas-lidar-dop-1_sterzing_wind_{datestr}.png'

print(f"Saving plot to: {out_png}")
plt.savefig(out_png, dpi=150)
plt.show()

# Create a separate figure for wind speed and direction
print("Creating wind speed and direction plot...")

# Calculate wind speed and direction if u and v components exist
if 'uwind' in DS.variables and 'vwind' in DS.variables:
    # Calculate wind speed
    wind_speed = np.sqrt(DS['uwind']**2 + DS['vwind']**2)
    
    # Calculate wind direction (meteorological convention: direction wind is coming FROM)
    wind_direction = np.arctan2(-DS['uwind'], -DS['vwind']) * 180 / np.pi
    # Convert to 0-360 degrees
    wind_direction = (wind_direction + 360) % 360
    
    # Create new figure for speed and direction
    fig2 = plt.figure(figsize=(14, 12))
    fig2.subplots_adjust(hspace=0.4, top=0.88, bottom=0.10)

    ax_speed = fig2.add_subplot(311)
    ax_dir = fig2.add_subplot(312)
    ax_speed_raw = fig2.add_subplot(313)
    
    # Set same y-limits as the component plots
    ax_speed.set_ylim(0, hmax)
    ax_dir.set_ylim(0, hmax)
    ax_speed_raw.set_ylim(0, hmax)
    
    # Format x-axis
    ax_speed.xaxis.set_major_formatter(myFmt)
    ax_dir.xaxis.set_major_formatter(myFmt)
    ax_speed_raw.xaxis.set_major_formatter(myFmt)
    
    # Plot wind speed
    h_speed = ax_speed.pcolormesh(
        dtime[:],
        rng,
        wind_speed[:, :].transpose(),
        vmin=0, vmax=20,
        cmap='viridis',
        shading='auto'
    )
    cb_speed = plt.colorbar(h_speed, ax=ax_speed, orientation='vertical')
    cb_speed.set_label('Wind Speed (m s$^{-1}$)')
    ax_speed.set_ylabel('Height (km AGL)')
    ax_speed.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_speed.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax_speed.set_xlim(plot_start, plot_end)
    ax_speed.set_title('Wind Speed')
    
    # Plot wind direction
    h_dir = ax_dir.pcolormesh(
        dtime[:],
        rng,
        wind_direction[:, :].transpose(),
        vmin=0, vmax=360,
        cmap='twilight_shifted',
        shading='auto'
    )
    cb_dir = plt.colorbar(h_dir, ax=ax_dir, orientation='vertical')
    cb_dir.set_label('Wind from Direction (degrees)')
    cb_dir.set_ticks([0, 90, 180, 270, 360])
    cb_dir.set_ticklabels(['N (0°)', 'E (90°)', 'S (180°)', 'W (270°)', 'N (360°)'])
    ax_dir.set_ylabel('Height (km AGL)')
    ax_dir.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_dir.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax_dir.set_xlim(plot_start, plot_end)
    ax_dir.set_title('Wind Direction')
    
    # Plot raw wind speed if raw components exist
    if 'uwind_raw' in DS.variables and 'vwind_raw' in DS.variables:
        wind_speed_raw = np.sqrt(DS['uwind_raw']**2 + DS['vwind_raw']**2)
        
        h_speed_raw = ax_speed_raw.pcolormesh(
            dtime[:],
            rng,
            wind_speed_raw[:, :].transpose(),
            vmin=0, vmax=20,
            cmap='viridis',
            shading='auto'
        )
        cb_speed_raw = plt.colorbar(h_speed_raw, ax=ax_speed_raw, orientation='vertical')
        cb_speed_raw.set_label('Raw Wind Speed (m s$^{-1}$)')
        ax_speed_raw.set_title('Raw Wind Speed')
    else:
        ax_speed_raw.text(0.5, 0.5, 'Raw wind components not found', 
                         ha='center', va='center', transform=ax_speed_raw.transAxes)
    
    ax_speed_raw.set_ylabel('Height (km AGL)')
    ax_speed_raw.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_speed_raw.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax_speed_raw.set_xlabel('Time (UTC)')
    ax_speed_raw.set_xlim(plot_start, plot_end)

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    
    # Calculate plot center for title
    axes2 = [ax_speed, ax_dir, ax_speed_raw]
    lefts2 = [ax.get_position().x0 for ax in axes2]
    rights2 = [ax.get_position().x1 for ax in axes2]
    plot_left2 = min(lefts2)
    plot_right2 = max(rights2)
    plot_center2 = plot_left2 + (plot_right2 - plot_left2) / 2

    # Now set the suptitle
    fig2.suptitle(
        f"NCAS Doppler Lidar 1, TEAMx Sterzing Wind Speed & Direction {title_date}{elevation_str}",
        fontsize=16,
        y=0.96,
        x=plot_center2
    )
    
    # Save second figure in the same subdirectory
    if elevation_str:
        elev_suffix = filename.split('elev')[1].split('_')[0]
        out_png2 = plot_subdir / f'ncas-lidar-dop-1_sterzing_wind_speed_dir_elev{elev_suffix}_{datestr}.png'
    else:
        out_png2 = plot_subdir / f'ncas-lidar-dop-1_sterzing_wind_speed_dir_{datestr}.png'
    
    print(f"Saving wind speed/direction plot to: {out_png2}")
    plt.savefig(out_png2, dpi=150)
    plt.show()
    
else:
    print("Cannot create wind speed/direction plot: uwind or vwind components not found")
