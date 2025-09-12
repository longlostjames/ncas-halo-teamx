#!/usr/bin/env python

import doppy
import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Process HALO DOPPY data.")
parser.add_argument("--date", required=True, help="Date string in YYYYMMDD format")
parser.add_argument("--elevation", type=float, default=None, help="Elevation angle to filter files (e.g., 5.0, 15.0)")
parser.add_argument("--combine-elevations", nargs='+', type=float, help="Combine multiple elevations (e.g., --combine-elevations 7 75)")
args = parser.parse_args()

datestr = args.date
elevation_filter = args.elevation
combine_elevations = args.combine_elevations

# Check for conflicting arguments
if elevation_filter is not None and combine_elevations is not None:
    print("ERROR: Cannot specify both --elevation and --combine-elevations")
    print("Use --elevation for single elevation processing")
    print("Use --combine-elevations for multi-elevation processing")
    exit(1)

teamx_halo_path = f'/gws/pw/j07/ncas_obs_vol1/amf/raw_data/ncas-lidar-dop-1/incoming/20250603_teamx/Proc/'
#teamx_halo_path = f'/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/Proc/'


yr = datestr[0:4]
mo = datestr[0:6]

datepath = os.path.join(teamx_halo_path, yr, mo, datestr)
os.chdir(datepath)

# Get all potential VAD files
all_VAD_files = [os.path.join(datepath, f) for f in glob.glob(f'User5_18_{datestr}_*.hpl')]

# Function to get elevation from file
def get_file_elevation(file):
    try:
        halo_data_list = doppy.raw.HaloHpl.from_srcs([file], overlapped_gates=True)
        lidar_data = halo_data_list[0]
        
        elevation = None
        if hasattr(lidar_data, 'elevation'):
            if hasattr(lidar_data.elevation, 'values'):
                elevation = float(lidar_data.elevation.values[0])
            else:
                elevation = float(lidar_data.elevation[0])
        elif hasattr(lidar_data, 'phi'):
            if hasattr(lidar_data.phi, 'values'):
                elevation = float(lidar_data.phi.values[0])
            else:
                elevation = float(lidar_data.phi[0])
        
        return elevation
    except Exception as e:
        return None

# Handle multi-elevation combination
if combine_elevations is not None:
    print(f"=== MULTI-ELEVATION PROCESSING: {combine_elevations} ===")
    
    # Analyze all files to group by elevation
    print("Analyzing files and grouping by elevation...")
    elevation_groups = {}
    
    for file in all_VAD_files:
        elevation = get_file_elevation(file)
        # Print file time (from filename or file header)
        try:
            # Example: extract time from filename (assuming format: User5_18_YYYYMMDD_HHMMSS.hpl)
            basename = os.path.basename(file)
            time_str = basename.split('_')[3].split('.')[0]  # HHMMSS
            print(f"Processing file: {basename}, time: {time_str}")
        except Exception:
            print(f"Processing file: {basename}, time: UNKNOWN")
        if elevation is not None:
            elev_rounded = round(elevation, 1)
            if elev_rounded not in elevation_groups:
                elevation_groups[elev_rounded] = []
            elevation_groups[elev_rounded].append(file)
    
    print(f"Found elevation groups: {sorted(elevation_groups.keys())}")
    
    # Check that all requested elevations exist
    missing_elevations = [e for e in combine_elevations if e not in elevation_groups]
    if missing_elevations:
        print(f"ERROR: Requested elevations not found: {missing_elevations}")
        print(f"Available elevations: {sorted(elevation_groups.keys())}")
        exit(1)
    
    # Process each elevation separately
    print("\nProcessing each elevation separately...")
    wind_results = []
    
    for elev in sorted(combine_elevations):
        print(f"\nProcessing elevation {elev}°...")
        elev_files = elevation_groups[elev]
        print(f"  Found {len(elev_files)} files for {elev}° elevation")

        print(f"Elev files {elev_files}")
        try:
            # Process this elevation
            wind_elev = doppy.product.Wind.from_halo_data(
                data=elev_files,
                options=doppy.product.wind.Options(azimuth_offset_deg=286.1, overlapped_gates=True),
            )
            
            wind_results.append((elev, wind_elev))
            print(f"  ✓ Successfully processed {elev}° elevation")
            
        except Exception as e:
            print(f"  ✗ Failed to process {elev}° elevation: {e}")
            continue
    
    if not wind_results:
        print("\nERROR: No elevations could be processed successfully!")
        exit(1)
    
    print(f"\nSuccessfully processed {len(wind_results)} elevation(s)")
    
    # Save each elevation separately and create a combined filename
    out_dir = "/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed"
    os.makedirs(out_dir, exist_ok=True)
    
    elevation_labels = []
    
    for elev, wind_data in wind_results:
        # Save individual elevation file
        elev_filename = os.path.join(out_dir, f"ncas-lidar-dop-1_{datestr}_wind_elev{elev:g}_v1.0.nc")
        print(f"Writing {elev}° elevation data to: {elev_filename}")
        wind_data.write_to_netcdf(elev_filename)
        elevation_labels.append(f"elev{elev:g}")
    
    # Create a combined label for the multi-elevation dataset
    combined_label = "_".join(elevation_labels)
    
    # For now, we'll save the first elevation as the "combined" result
    # (True combination would require merging the datasets, which is complex)
    if len(wind_results) == 1:
        combined_filename = os.path.join(out_dir, f"ncas-lidar-dop-1_{datestr}_wind_{combined_label}_v1.0.nc")
        print(f"Creating combined file: {combined_filename}")
        wind_results[0][1].write_to_netcdf(combined_filename)
    else:
        print(f"\nNOTE: Individual elevation files created.")
        print(f"True temporal combination of different elevations requires")
        print(f"additional processing beyond doppy's standard capabilities.")
        print(f"Use the individual elevation files or process them separately.")
    
    print("Multi-elevation processing complete!")
    exit(0)

# Original single-elevation processing below
# Filter by elevation if specified
if elevation_filter is not None:
    print(f"Filtering files for elevation angle: {elevation_filter}°")
    VAD_files = []
    
    for file in all_VAD_files:
        try:
            # Load file to check elevation
            halo_data_list = doppy.raw.HaloHpl.from_srcs([file], overlapped_gates=True)
            lidar_data = halo_data_list[0]
            
            # Try to get elevation from the data
            elevation = None
            if hasattr(lidar_data, 'elevation'):
                if hasattr(lidar_data.elevation, 'values'):
                    elevation = float(lidar_data.elevation.values[0])
                else:
                    elevation = float(lidar_data.elevation[0])
            elif hasattr(lidar_data, 'phi'):  # Sometimes elevation is stored as phi
                if hasattr(lidar_data.phi, 'values'):
                    elevation = float(lidar_data.phi.values[0])
                else:
                    elevation = float(lidar_data.phi[0])
        
            # Print file time (from filename)
            basename = os.path.basename(file)
            try:
                time_str = basename.split('_')[3].split('.')[0]  # HHMMSS
                print(f"Processing file: {basename}, time: {time_str}")
            except Exception:
                print(f"Processing file: {basename}, time: UNKNOWN")
            
            # Check if elevation matches (with small tolerance for floating point comparison)
            if elevation is not None and abs(elevation - elevation_filter) < 0.1:
                VAD_files.append(file)
                print(f"  Including: {os.path.basename(file)} (elevation: {elevation:.1f}°)")
            else:
                print(f"  Skipping: {os.path.basename(file)} (elevation: {elevation:.1f}° != {elevation_filter}°)")
                
        except Exception as e:
            print(f"  Error checking elevation for {os.path.basename(file)}: {e}")
            continue
    
    print(f"Selected {len(VAD_files)} files out of {len(all_VAD_files)} total files")
else:
    VAD_files = all_VAD_files
    print(f"No elevation filter specified, using all {len(VAD_files)} VAD files")

background_files = [os.path.join(datepath, f) for f in glob.glob(f'Background_*.txt')]

print(f"VAD files: {len(VAD_files)}")
print(f"Background files: {len(background_files)}")

if not VAD_files:
    print("No VAD files found matching criteria. Exiting.")
    exit(1)

# Continue with the original DEBUG analysis and processing...
print("\nDEBUG: Analyzing selected VAD files for wind processing compatibility...")

try:
    # Load all selected files to analyze them
    halo_data_list = doppy.raw.HaloHpl.from_srcs(VAD_files, overlapped_gates=True)
    
    print(f"Loaded {len(halo_data_list)} data objects")
    
    # Analyze each data object
    elevations = []
    azimuth_counts = []
    scan_types = []
    
    for i, data in enumerate(halo_data_list):
        # Get elevation
        elevation = None
        if hasattr(data, 'elevation'):
            if hasattr(data.elevation, 'values'):
                elevation = float(data.elevation.values[0]) if len(data.elevation.values) > 0 else None
            else:
                elevation = float(data.elevation[0]) if len(data.elevation) > 0 else None
        elif hasattr(data, 'phi'):
            if hasattr(data.phi, 'values'):
                elevation = float(data.phi.values[0]) if len(data.phi.values) > 0 else None
            else:
                elevation = float(data.phi[0]) if len(data.phi) > 0 else None
        
        if elevation is not None:
            elevations.append(elevation)
        
        # Get azimuth information
        azimuth_angles = None
        if hasattr(data, 'azimuth'):
            if hasattr(data.azimuth, 'values'):
                azimuth_angles = data.azimuth.values
            else:
                azimuth_angles = data.azimuth
        elif hasattr(data, 'theta'):
            if hasattr(data.theta, 'values'):
                azimuth_angles = data.theta.values
            else:
                azimuth_angles = data.theta
        
        if azimuth_angles is not None:
            unique_azimuths = len(np.unique(np.round(azimuth_angles, 1)))
            azimuth_counts.append(unique_azimuths)
        
        # Get scan type if available
        scan_type = "unknown"
        if hasattr(data, 'scan_type'):
            scan_type = str(data.scan_type)
        elif hasattr(data.header, 'scan_type'):
            scan_type = str(data.header.scan_type)
        
        scan_types.append(scan_type)
        
        print(f"  File {i+1}: elevation={elevation:.1f}°, azimuth_count={azimuth_counts[-1] if azimuth_counts else 'unknown'}, scan_type={scan_type}")
    
    # Check doppy wind processing requirements
    print(f"\nDOPPY Wind Processing Requirements Check:")
    print(f"  Elevation angles found: {set(elevations)}")
    print(f"  Multiple elevations: {len(set(elevations)) > 1}")
    print(f"  Elevation >= 80°: {any(e >= 80 for e in elevations)}")
    print(f"  Elevation <= 4°: {any(e <= 4 for e in elevations)}")
    print(f"  Azimuth counts: {azimuth_counts}")
    print(f"  Any scan with <= 3 azimuths: {any(c <= 3 for c in azimuth_counts if c is not None)}")
    
    # Identify the issue
    issues = []
    if len(set(elevations)) > 1:
        issues.append(f"Multiple elevation angles: {set(elevations)}")
    if any(e >= 80 for e in elevations):
        issues.append(f"Elevation >= 80°: {[e for e in elevations if e >= 80]}")
    if any(e <= 4 for e in elevations):
        issues.append(f"Elevation <= 4°: {[e for e in elevations if e <= 4]}")
    if any(c <= 3 for c in azimuth_counts if c is not None):
        issues.append(f"Scans with <= 3 azimuths: {[c for c in azimuth_counts if c is not None and c <= 3]}")
    
    if issues:
        print(f"\nISSUES PREVENTING WIND PROCESSING:")
        for issue in issues:
            print(f"  - {issue}")
        
        # Suggest solutions
        print(f"\nSUGGESTED SOLUTIONS:")
        if len(set(elevations)) > 1:
            print(f"  - Use --elevation filter to select a single elevation angle")
            print(f"  - Available elevations: {sorted(set(elevations))}")
            print(f"  - Or use --combine-elevations to process multiple elevations separately")
        if any(e <= 4 for e in elevations):
            print(f"  - Avoid elevations <= 4° (too low for VAD processing)")
        if any(e >= 80 for e in elevations):
            print(f"  - Avoid elevations >= 80° (too high for VAD processing)")
        if any(c <= 3 for c in azimuth_counts if c is not None):
            print(f"  - Check that scans have enough azimuth angles (need > 3)")
        
        print(f"\nRecommended commands:")
        good_elevations = [e for e in set(elevations) if 4 < e < 80]
        if good_elevations:
            print(f"  Single elevation: python {os.path.basename(__file__)} --date {datestr} --elevation {sorted(good_elevations)[0]}")
            if len(good_elevations) > 1:
                elev_list = " ".join([str(e) for e in sorted(good_elevations)])
                print(f"  Multi elevation: python {os.path.basename(__file__)} --date {datestr} --combine-elevations {elev_list}")
        
        exit(1)
    
except Exception as e:
    print(f"Error analyzing VAD files: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# If we get here, the files should be compatible
print(f"\nFiles appear compatible with doppy wind processing. Proceeding...")

# You can also pass instrument azimuth offset in degrees as an option
wind = doppy.product.Wind.from_halo_data(
    data=VAD_files,
    options=doppy.product.wind.Options(azimuth_offset_deg=286.1, overlapped_gates=True),
)

# Construct output directory and filename
out_dir = "/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed"
os.makedirs(out_dir, exist_ok=True)

# Include elevation in filename if specified
if elevation_filter is not None:
    out_nc = os.path.join(out_dir, f"ncas-lidar-dop-1_{datestr}_wind_elev{elevation_filter:g}_v1.0.nc")
else:
    out_nc = os.path.join(out_dir, f"ncas-lidar-dop-1_{datestr}_wind_v1.0.nc")

print(f"Writing output to: {out_nc}")
wind.write_to_netcdf(out_nc)
print("Processing complete!")

