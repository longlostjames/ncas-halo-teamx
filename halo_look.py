import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def read_hpl(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find header and column indices
    header_lines = []
    data_start_idx = 0
    col_names = None  # Add this before the loop that sets col_names

    with open(filename) as f:
        for idx, line in enumerate(f):
            if line.startswith('##'):
                header_lines.append(line.strip())
            elif line.startswith('#'):
                if 'column names' in line.lower():
                    col_names = line.strip('#').strip().split()
            else:
                data_start_idx = idx
                break

    if col_names is None:
        raise ValueError("Column names not found in the file.")

    # Parse metadata (e.g., range gates, site info)
    metadata = {}
    for line in header_lines:
        if '=' in line:
            key, value = line.lstrip('#').split('=', 1)
            metadata[key.strip()] = value.strip()

    # Load the data
    df = pd.read_csv(filename, delim_whitespace=True, skiprows=data_start_idx, names=col_names)

    # Convert timestamp
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f')
    
    return df, metadata

def plot_los_velocity(df, metadata):
    # Extract parameters
    n_ranges = int(metadata.get('nRanges', 0))
    range_resolution = float(metadata.get('dRange', 30))  # in meters

    # Extract profile count
    profiles = df['Time'].unique()
    n_profiles = len(profiles)

    # Build time-height matrix
    vel_cols = [col for col in df.columns if col.startswith('Velocity_')]
    V = df[vel_cols].to_numpy().reshape((n_profiles, n_ranges))

    # Range in meters
    ranges = np.arange(n_ranges) * range_resolution

    # Plot
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(profiles, ranges, V.T, shading='auto', cmap='RdBu_r')
    plt.colorbar(label='LOS Velocity (m/s)')
    plt.xlabel('Time')
    plt.ylabel('Range (m)')
    plt.title('Halo Streamline LOS Velocity')
    plt.tight_layout()
    plt.show()
