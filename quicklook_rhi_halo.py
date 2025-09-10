#!/usr/bin/env python3
"""
Quicklook script for HALO RHI scans
Creates range-height plots for RHI data
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import cmocean
import pyart
import doppy.raw
from pathlib import Path

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
        ncas_logo_h = 0.08  # Doubled from 0.05
        ncas_logo_w = ncas_logo_h * (ncas_logo_img.shape[1] / ncas_logo_img.shape[0])
        ncas_logo_ax = fig.add_axes([1.0 - ncas_logo_w - 0.01, 0.94, ncas_logo_w, ncas_logo_h])
        ncas_logo_ax.imshow(ncas_logo_img)
        ncas_logo_ax.axis('off')

def process_single_file(selected_file, file_index, total_files):
    """Process a single RHI file and create range-height plot"""
    
    print(f"Processing file {file_index+1}/{total_files}: {os.path.basename(selected_file)}")
    
    # Check scan type in HPL file header
    try:
        with open(selected_file, 'r') as f:
            # Read lines to find scan type and azimuth angle
            scan_type = None
            azimuth_angle = None
            lines = f.readlines()
            
            for i, line in enumerate(lines):
                if i > 50:  # Stop after first 50 lines
                    break
                    
                # Look for scan type
                if line.startswith('Scan type'):
                    scan_type = line.strip()
                    print(f"Found scan type: {scan_type}")
                
                # Look for the data separator line "****"
                if line.strip() == '****':
                    # The next line should be the first data line
                    if i + 1 < len(lines):
                        data_line = lines[i + 1].strip()
                        # Parse: Decimal time, Azimuth, Elevation, Pitch, Roll
                        # Format: 9.18540833 353.00  -0.00 0.05 -0.01
                        try:
                            parts = data_line.split()
                            if len(parts) >= 5:
                                raw_azimuth = float(parts[1])  # Second column is azimuth
                                
                                # Apply azimuth offset of 286.1 degrees
                                azimuth_offset = 286.1
                                azimuth_angle = (raw_azimuth + azimuth_offset) % 360.0
                                
                                print(f"Raw RHI azimuth: {raw_azimuth}°")
                                print(f"Corrected RHI azimuth: {azimuth_angle}°")
                            else:
                                print(f"Unexpected data line format: {data_line}")
                        except (ValueError, IndexError) as e:
                            print(f"Could not parse azimuth from data line: {data_line}, error: {e}")
                    break
            
            if scan_type is None:
                print("No 'Scan type' line found in file header")
                return None
            
            # Check if scan type contains RHI
            if 'RHI' not in scan_type.upper():
                print(f"Skipping file - scan type does not contain RHI: {scan_type}")
                return None
            
            print(f"Valid RHI scan type found: {scan_type}")
            
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
        elevation = halo_data.elevation if hasattr(halo_data.elevation, 'shape') else halo_data.elevation.values
        time_vals = halo_data.time if hasattr(halo_data.time, 'shape') else halo_data.time.values
        
        print(f"Elevation range: {elevation.min():.1f} to {elevation.max():.1f} degrees")
        
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
    
    # Convert range to meters for PyART
    range_m = range_vals * 1000.0
    
    # Create proper PyART radar object for RHI
    fields = {}
    
    # Add velocity field if available
    if velocity is not None:
        if len(velocity.shape) == 2:
            vel_data = velocity  # Data is [elevation, range]
        elif len(velocity.shape) == 3:
            vel_data = velocity[0, :, :]  # Use first time step
        else:
            vel_data = None
            
        if vel_data is not None:
            fields['velocity'] = {
                'data': np.ma.masked_invalid(vel_data),
                'units': 'm/s',
                'long_name': 'Doppler Velocity',
                'standard_name': 'radial_velocity_of_scatterers_away_from_instrument'
            }
    
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
        
        # Create SNR mask (True where SNR < -15 dB, these will be masked out)
        snr_mask = snr_db < -19.5
        print(f"SNR masking: {np.sum(snr_mask)} / {snr_mask.size} points masked (SNR < -19 dB)")

        fields['snr'] = {
            'data': np.ma.masked_invalid(snr_db),
            'units': 'dB',
            'long_name': 'Signal-to-Noise Ratio',
            'standard_name': 'snr'
        }
    
    # Apply SNR mask to all fields
    if snr_mask is not None:
        for field_name, field_dict in fields.items():
            if field_name != 'snr':  # Don't mask the SNR field itself
                # Apply additional SNR masking to existing data
                masked_data = np.ma.array(field_dict['data'], mask=field_dict['data'].mask)
                masked_data.mask = np.ma.mask_or(masked_data.mask, snr_mask)
                field_dict['data'] = masked_data
                print(f"Applied SNR mask to {field_name} field")
    
    # Create proper radar object using pyart.testing.make_empty_rhi_radar
    radar = pyart.testing.make_empty_rhi_radar(
        ngates=len(range_m),
        rays_per_sweep=len(elevation),
        nsweeps=1
    )
    
    # Update radar object with our data
    radar.range['data'] = range_m.astype(np.float32)
    radar.elevation['data'] = elevation.astype(np.float32)
    radar.azimuth['data'] = np.full_like(elevation, azimuth_angle if azimuth_angle else 0.0, dtype=np.float32)
    radar.time['data'] = np.arange(len(elevation), dtype=np.float32)
    
    # Add fields to radar object
    radar.fields = fields
    
    # Create figure with PyART
    fig = plt.figure(figsize=(16, 12))
    
    # Count available fields to determine subplot layout
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
        
        display.plot_rhi(field_name, 0, ax=ax,
                        title=title,
                        colorbar_label=colorbar_label,
                        colorbar_flag=True,
                        colorbar_orient='horizontal',
                        **plot_kwargs)
        
        # Set appropriate limits for RHI (range vs height)
        display.set_limits(xlim=(0, 3), ylim=(0, 3), ax=ax)  # 0-3km range, 0-3km height
        ax.set_aspect('equal')  # Set equal aspect ratio
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Range (km)')
        ax.set_ylabel('Height (km)')
    
    print(f"Created {len(available_fields)} RHI plots: {[field[0] for field in available_fields]}")
    
    # Add logos and title
    add_logos(fig, 0.5)
    azimuth_str = f"{azimuth_angle:.1f}°" if azimuth_angle else "Unknown"
    fig.suptitle(
        f"NCAS Doppler Lidar 1, TEAMx Sterzing RHI Scan\nAzimuth: {azimuth_str}, {file_timestamp_title}",
        fontsize=16,
        y=0.95  # Higher position with more room
    )
    
    # Save figure
    azimuth_str_filename = f"{azimuth_angle:.1f}deg" if azimuth_angle else "Unknown"
    out_png = quicklook_dir / f'ncas-lidar-dop-1_sterzing_rhi_{azimuth_str_filename.replace(".", "p")}_{datestr}-{file_timestamp_filename}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close(fig)
    
    return out_png

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create quicklook plots for HALO RHI data')
parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
parser.add_argument('--max_files', type=int, help='Maximum number of files to process')

args = parser.parse_args()

# Parse date
datestr = args.date
max_files = args.max_files

yr = datestr[0:4]
mo = datestr[0:6]

# Set up paths
teamx_halo_path = "/gws/pw/j07/ncas_obs_vol1/amf/raw_data/ncas-lidar-dop-1/incoming/20250603_teamx/Proc"
datepath = os.path.join(teamx_halo_path, yr, mo, datestr)

# Set up output directory
base_out_dir = Path("/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed")
quicklook_dir = base_out_dir / "quicklooks" / "rhi" / datestr
quicklook_dir.mkdir(parents=True, exist_ok=True)

# Find RHI files
os.chdir(datepath)
RHI_files = [os.path.join(datepath, f) for f in glob.glob(f'User5_18_{datestr}_*.hpl')]

print(f"Found {len(RHI_files)} potential RHI files for {datestr}")

if not RHI_files:
    print(f"No files found for date {datestr}")
    print(f"Searched in: {datepath}")
    sys.exit(1)

# Limit number of files if specified
if max_files is not None and len(RHI_files) > max_files:
    RHI_files = RHI_files[:max_files]
    print(f"Processing first {max_files} files only")

# Sort files to ensure chronological order
RHI_files.sort()

# Process each file
successful_plots = 0
failed_plots = 0

for file_index, rhi_file in enumerate(RHI_files):
    try:
        result = process_single_file(rhi_file, file_index, len(RHI_files))
        if result is not None:
            successful_plots += 1
            print(f"Successfully created: {result}")
        else:
            failed_plots += 1
            print(f"Skipped file: {rhi_file}")
    except Exception as e:
        failed_plots += 1
        print(f"Error processing file {rhi_file}: {e}")

print(f"\nProcessing complete!")
print(f"Successfully created {successful_plots} RHI plots")
print(f"Skipped {failed_plots} files (non-RHI or errors)")
print(f"Output directory: {quicklook_dir}")