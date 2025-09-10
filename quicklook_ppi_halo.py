#!/usr/bin/env python

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import matplotlib.dates as mdates
import argparse
import sys
import os
import glob
import cftime
import cmocean
import pandas as pd
import pyart
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import doppy

# Parse date from command line
parser = argparse.ArgumentParser(description="Quicklook polar plots for HALO VAD/PPI data.")
parser.add_argument("--date", required=True, help="Date string in YYYYMMDD format")
parser.add_argument("--max_files", default=None, type=int, help="Maximum number of files to process (default: all)")
args = parser.parse_args()

datestr = args.date
max_files = args.max_files

# Set path for HALO VAD files (actually PPI scans at 5deg elevation)
teamx_halo_path = f'/gws/pw/j07/ncas_obs_vol1/amf/raw_data/ncas-lidar-dop-1/incoming/20250603_teamx/Proc/'

yr = datestr[0:4]
mo = datestr[0:6]

datepath = os.path.join(teamx_halo_path, yr, mo, datestr)
os.chdir(datepath)

VAD_files = [os.path.join(datepath, f) for f in glob.glob(f'User5_18_{datestr}_*.hpl')]
background_files = [os.path.join(datepath, f) for f in glob.glob(f'Background_*.txt')]

print(f"Found {len(VAD_files)} VAD/PPI files for {datestr}")

if not VAD_files:
    print(f"No VAD files found for date {datestr}")
    print(f"Searched in: {datepath}")
    sys.exit(1)

# Limit number of files if specified
if max_files is not None and len(VAD_files) > max_files:
    VAD_files = VAD_files[:max_files]
    print(f"Processing first {max_files} files only")

# Sort files to ensure chronological order
VAD_files.sort()

# Set up output directory
base_out_dir = Path("/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed")
quicklook_dir = base_out_dir / "quicklooks" / "ppi" / datestr
quicklook_dir.mkdir(parents=True, exist_ok=True)

def add_logos(fig, plot_center):
    """Add AMOF and NCAS logos to figure"""
    # Add AMOF logo as an inset axis in the bottom left outside the plot
    logo_path = "/home/users/cjwalden/git/halo-teamx/amof-web-header-wr.png"
    if os.path.exists(logo_path):
        logo_img = mpimg.imread(logo_path)
        logo_h = 0.06  
        logo_w = logo_h * (logo_img.shape[1] / logo_img.shape[0])
        logo_ax = fig.add_axes([0.01, 0.01, logo_w, logo_h])
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')

    # Add NCAS logo as an inset axis in the top right
    ncas_logo_path = "/home/users/cjwalden/git/halo-teamx/NCAS_national_centre_logo_transparent.png"
    if os.path.exists(ncas_logo_path):
        ncas_logo_img = mpimg.imread(ncas_logo_path)
        ncas_logo_h = 0.08 
        ncas_logo_w = ncas_logo_h * (ncas_logo_img.shape[1] / ncas_logo_img.shape[0])
        ncas_logo_ax = fig.add_axes([1.0 - ncas_logo_w - 0.01, 0.94, ncas_logo_w, ncas_logo_h])
        ncas_logo_ax.imshow(ncas_logo_img)
        ncas_logo_ax.axis('off')

def process_single_file(selected_file, file_index, total_files):
    """Process a single VAD/PPI file and create polar plot"""
    
    print(f"Processing file {file_index+1}/{total_files}: {os.path.basename(selected_file)}")
    
    # Check scan type in HPL file header
    try:
        with open(selected_file, 'r') as f:
            # Read first few lines to find scan type
            scan_type = None
            for i, line in enumerate(f):
                if i > 20:  # Stop after first 20 lines (header should be within this)
                    break
                if line.startswith('Scan type'):
                    scan_type = line.strip()
                    print(f"Found scan type: {scan_type}")
                    break
            
            if scan_type is None:
                print("No 'Scan type' line found in file header")
                return None
            
            # Check if scan type contains VAD or PPI
            if 'VAD' not in scan_type.upper() and 'PPI' not in scan_type.upper():
                print(f"Skipping file - scan type does not contain VAD or PPI: {scan_type}")
                return None
            
            print(f"Valid scan type found: {scan_type}")
            
    except Exception as e:
        print(f"Error reading file header: {e}")
        return None
    
    # Load single .hpl file using doppy's raw data loader
    try:
        # Use HaloHpl.from_srcs method
        halo_data_list = doppy.raw.HaloHpl.from_srcs(
            [selected_file], 
            overlapped_gates=True
        )
        
        # HaloHpl.from_srcs returns a list, so get the first element
        halo_data = halo_data_list[0] if isinstance(halo_data_list, list) else halo_data_list
        
        # Use radial_distance attribute from HALO data
        if hasattr(halo_data, 'radial_distance'):
            range_vals = halo_data.radial_distance / 1000.0  # Convert to km
        elif hasattr(halo_data, 'range'):
            range_vals = halo_data.range / 1000.0  # Convert to km
        else:
            # Fallback: Create default range gates if not found
            range_gate_indices = np.arange(200)  # Typical HALO range gate count
            range_gate_length = 30.0  # meters per range gate
            range_offset = 0.0  # offset to first range gate
            range_vals = (range_gate_indices * range_gate_length + range_offset) / 1000.0
        
        # Convert to numpy array if it's not already
        if hasattr(range_vals, 'values'):
            range_vals = range_vals.values
        
        # Get other data arrays
        azimuth = halo_data.azimuth if hasattr(halo_data.azimuth, 'shape') else halo_data.azimuth.values
        time_vals = halo_data.time if hasattr(halo_data.time, 'shape') else halo_data.time.values
        
        # Apply azimuth offset of 286.1 degrees
        azimuth_offset = 286.1
        azimuth_corrected = (azimuth + azimuth_offset) % 360.0
        
        print(f"Original azimuth range: {azimuth.min():.1f} to {azimuth.max():.1f} degrees")
        print(f"Corrected azimuth range: {azimuth_corrected.min():.1f} to {azimuth_corrected.max():.1f} degrees")
        
        # Get measurement variables
        velocity = None
        if hasattr(halo_data, 'radial_velocity'):
            velocity = halo_data.radial_velocity if hasattr(halo_data.radial_velocity, 'shape') else halo_data.radial_velocity.values
            print(f"Radial velocity data found: shape = {velocity.shape}")
        elif hasattr(halo_data, 'v'):
            velocity = halo_data.v if hasattr(halo_data.v, 'shape') else halo_data.v.values
            print(f"Velocity data found: shape = {velocity.shape}")
        else:
            print("No 'radial_velocity' or 'v' attribute found in halo_data")
            print(f"Available attributes: {[attr for attr in dir(halo_data) if not attr.startswith('_')]}")
        
        intensity = None
        if hasattr(halo_data, 'intensity'):
            intensity = halo_data.intensity if hasattr(halo_data.intensity, 'shape') else halo_data.intensity.values
            print(f"Intensity data found: shape = {intensity.shape}")
        else:
            print("No 'intensity' attribute found")
        
        beta = None
        if hasattr(halo_data, 'beta'):
            beta = halo_data.beta if hasattr(halo_data.beta, 'shape') else halo_data.beta.values
            print(f"Beta data found: shape = {beta.shape}")
        else:
            print("No 'beta' attribute found")
    
    except Exception as e:
        print(f"Error loading raw HALO data: {e}")
        return None
    
    # Get file timestamp for title and filename
    file_timestamp_title = pd.to_datetime(time_vals[0]).strftime("%d-%b-%Y %H:%M UTC")
    file_timestamp_filename = pd.to_datetime(time_vals[0]).strftime("%H%M%S")
    
    # Convert range to meters for PyART (PyART expects meters)
    range_m = range_vals * 1000.0
    
    # Create proper PyART radar object
    # Start with required fields
    fields = {}
    
    # Add velocity field if available
    if velocity is not None:
        print(f"Processing velocity data with shape: {velocity.shape}")
        if len(velocity.shape) == 2:
            vel_data = velocity  # Data is [azimuth, range]
            print("Using 2D velocity data directly")
        elif len(velocity.shape) == 3:
            vel_data = velocity[0, :, :]  # Use first time step
            print("Using first time step of 3D velocity data")
        else:
            vel_data = None
            print(f"Unexpected velocity shape: {velocity.shape}")
            
        if vel_data is not None:
            print(f"Adding velocity field to radar object")
            fields['velocity'] = {
                'data': np.ma.masked_invalid(vel_data),
                'units': 'm/s',
                'long_name': 'Doppler Velocity',
                'standard_name': 'radial_velocity_of_scatterers_away_from_instrument'
            }
        else:
            print("vel_data is None, not adding velocity field")
    else:
        print("velocity is None, skipping velocity field")
    
    # Add intensity field if available
    if intensity is not None:
        if len(intensity.shape) == 2:
            int_data = intensity
        elif len(intensity.shape) == 3:
            int_data = intensity[0, :, :]
        else:
            int_data = None
            
        if int_data is not None:
            fields['intensity'] = {
                'data': np.ma.masked_invalid(int_data),
                'units': 'SNR+1',
                'long_name': 'Intensity',
                'standard_name': 'intensity'
            }
    
    # Add beta field if available
    if beta is not None:
        if len(beta.shape) == 2:
            beta_data = beta
        elif len(beta.shape) == 3:
            beta_data = beta[0, :, :]
        else:
            beta_data = None
            
        if beta_data is not None:
            fields['beta'] = {
                'data': np.ma.masked_invalid(beta_data),
                'units': 'm-1 sr-1',
                'long_name': 'Attenuated Backscatter',
                'standard_name': 'backscatter'
            }
    
    # Calculate SNR from intensity and create mask
    snr_mask = None
    if intensity is not None and int_data is not None:
        snr_data = int_data - 1
        snr_db = 10 * np.log10(np.maximum(snr_data, 0.01))
        
        # Create SNR mask (True where SNR < -19.5 dB, these will be masked out)
        snr_mask = snr_db < -19.5
        print(f"SNR masking: {np.sum(snr_mask)} / {snr_mask.size} points masked (SNR < -19.5 dB)")
        
        fields['snr'] = {
            'data': np.ma.masked_invalid(snr_db),
            'units': 'dB',
            'long_name': 'Signal-to-Noise Ratio',
            'standard_name': 'snr'
        }
    
    # Apply SNR mask to all fields except SNR itself
    if snr_mask is not None:
        for field_name, field_dict in fields.items():
            if field_name != 'snr':  # Don't mask the SNR field itself
                # Apply additional SNR masking to existing data
                masked_data = np.ma.array(field_dict['data'], mask=field_dict['data'].mask)
                masked_data.mask = np.ma.mask_or(masked_data.mask, snr_mask)
                field_dict['data'] = masked_data
                print(f"Applied SNR mask to {field_name} field")
    
    # Create proper radar object using pyart.testing.make_empty_ppi_radar
    radar = pyart.testing.make_empty_ppi_radar(
        ngates=len(range_m),
        rays_per_sweep=len(azimuth_corrected),  # Use corrected azimuth
        nsweeps=1
    )
    
    # Update radar object with our data
    radar.range['data'] = range_m.astype(np.float32)
    radar.azimuth['data'] = azimuth_corrected.astype(np.float32)  # Use corrected azimuth
    radar.elevation['data'] = np.full_like(azimuth_corrected, 5.0, dtype=np.float32)  # Use corrected azimuth
    radar.time['data'] = np.arange(len(azimuth_corrected), dtype=np.float32)  # Use corrected azimuth
    
    # Add fields to radar object
    radar.fields = fields
    
    # Create figure with PyART
    fig = plt.figure(figsize=(16, 12))
    
    # Count available fields to determine subplot layout (remove intensity)
    available_fields = []
    if 'velocity' in radar.fields:
        available_fields.append(('velocity', 'Doppler Velocity', 'Radial Velocity (m/s)', 
                               {'vmin': -15, 'vmax': 15, 'cmap': cmocean.cm.balance}))
    
    if 'beta' in radar.fields:
        available_fields.append(('beta', 'Attenuated Backscatter', 'Backscatter (m⁻¹ sr⁻¹)', 
                               {'norm': colors.LogNorm(vmin=1e-8, vmax=1e-4), 'cmap': 'plasma'}))
    
    if 'snr' in radar.fields:
        available_fields.append(('snr', 'Signal-to-Noise Ratio', 'SNR (dB)', 
                               {'vmin': -20, 'vmax': 20, 'cmap': 'viridis'}))
    
    # If no fields available, skip this file
    if not available_fields:
        print("No valid data fields found, skipping file")
        plt.close(fig)
        return None
    
    # Determine subplot layout based on number of fields
    n_fields = len(available_fields)
    if n_fields == 1:
        rows, cols = 1, 1
        figsize = (8, 10)  # Made taller
    elif n_fields == 2:
        rows, cols = 1, 2
        figsize = (16, 10)  # Made taller
    else:  # 3 fields
        rows, cols = 1, 3
        figsize = (18, 8)   # Made taller
    
    # Recreate figure with appropriate size and extra space for horizontal colorbars
    plt.close(fig)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.80, bottom=0.12)  # Less space at bottom
    
    # Create plots for available fields
    for i, (field_name, title, colorbar_label, plot_kwargs) in enumerate(available_fields):
        ax = fig.add_subplot(rows, cols, i + 1)
        display = pyart.graph.RadarDisplay(radar)
        display.plot_ppi(field_name, 0, ax=ax,
                        title=title,
                        colorbar_label=colorbar_label,
                        colorbar_flag=True,
                        colorbar_orient='horizontal',
                        **plot_kwargs)
        display.set_limits(xlim=(-2, 2), ylim=(-2, 2), ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Update axis labels to use "lidar" instead of "radar"
        ax.set_xlabel('East-West Distance from Lidar (km)')
        ax.set_ylabel('North-South Distance from Lidar (km)')
    
    print(f"Created {len(available_fields)} plots: {[field[0] for field in available_fields]}")
    
    # Add logos and title
    add_logos(fig, 0.5)
    fig.suptitle(
        f"NCAS Doppler Lidar 1, TEAMx Sterzing VAD/PPI 5° Elevation\n{file_timestamp_title}",
        fontsize=16,
        y=0.95
    )
    
    # Save figure
    out_png = quicklook_dir / f'ncas-lidar-dop-1_sterzing_vad_ppi_5deg_{datestr}-{file_timestamp_filename}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close(fig)
    
    return out_png

# =============================================================================
# Main processing loop
# =============================================================================
print(f"Processing {len(VAD_files)} VAD/PPI files...")

successful_plots = 0
failed_plots = 0

for file_index, selected_file in enumerate(VAD_files):
    try:
        out_png = process_single_file(selected_file, file_index, len(VAD_files))
        if out_png:
            successful_plots += 1
            if file_index % 10 == 0:  # Progress update every 10 files
                print(f"Progress: {file_index+1}/{len(VAD_files)} files processed")
        else:
            failed_plots += 1
    except Exception as e:
        print(f"Error processing file {file_index}: {e}")
        failed_plots += 1
        continue

print(f"\nProcessing complete!")
print(f"Successfully created {successful_plots} plots")
print(f"Failed to process {failed_plots} files")
print(f"Output directory: {base_out_dir / 'quicklooks' / yr}")