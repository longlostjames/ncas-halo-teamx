#!/usr/bin/env python
"""
Create PPI map plots for HALO lidar data overlaid on geographic maps
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
import doppy
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import time
import requests
import math
import contextily as ctx

# Lidar location (Sterzing, Italy)
LIDAR_LAT = 46.885056
LIDAR_LON = 11.439319

def load_tiles_with_retry(ax, tile_source, zoom_level, max_retries=3, retry_delay=2):
    """
    Load map tiles with retry logic for handling temporary service outages
    """
    for attempt in range(max_retries):
        try:
            ax.add_image(tile_source, zoom_level)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Tile loading attempt {attempt + 1}/{max_retries} failed: {e}")
                print(f"Retrying in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                print(f"All {max_retries} tile loading attempts failed: {e}")
                return False
    return False

def add_logos(fig, plot_center):
    """Add AMOF and NCAS logos to figure"""
    # Add AMOF logo as an inset axis in the bottom left outside the plot
    logo_path = "/home/users/cjwalden/git/halo-teamx/amof-web-header-wr.png"
    if os.path.exists(logo_path):
        logo_img = mpimg.imread(logo_path)
        logo_h = 0.04
        logo_w = logo_h * (logo_img.shape[1] / logo_img.shape[0])
        logo_ax = fig.add_axes([0.01, 0.01, logo_w, logo_h])
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')

    # Move NCAS logo to the left a bit (e.g., from 0.94 to 0.90)
    ncas_logo_path = "/home/users/cjwalden/git/halo-teamx/NCAS_national_centre_logo_transparent.png"
    if os.path.exists(ncas_logo_path):
        ncas_logo_img = mpimg.imread(ncas_logo_path)
        ncas_logo_h = 0.05
        ncas_logo_w = ncas_logo_h * (ncas_logo_img.shape[1] / ncas_logo_img.shape[0])
        # Change 0.94 to 0.90 to move left
        ncas_logo_ax = fig.add_axes([1.0 - ncas_logo_w - 0.01, 0.90, ncas_logo_w, ncas_logo_h])
        ncas_logo_ax.imshow(ncas_logo_img)
        ncas_logo_ax.axis('off')

def create_ppi_map_plot(radar, field_name='velocity', max_range_km=2.0, timestamp_str="", elevation=None):
    """
    Create a PPI map plot using the native projection of the map tiles
    """
    
    # Calculate approximate extent in degrees
    lat_extent = max_range_km / 111.0
    lon_extent = max_range_km / (111.0 * np.cos(np.radians(LIDAR_LAT)))
    
    # Use Web Mercator projection (native for most tile services)
    proj = ccrs.epsg(3857)  # Web Mercator
    data_crs = ccrs.PlateCarree()  # For our lat/lon data
    
    # Create figure and axis
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=proj)
    
    # Transform extent to Web Mercator
    extent_lonlat = [LIDAR_LON - lon_extent, LIDAR_LON + lon_extent,
                     LIDAR_LAT - lat_extent, LIDAR_LAT + lat_extent]
    
    ax.set_extent(extent_lonlat, crs=data_crs)
    
    # Try to add map tiles with retry logic
    try:
        # Option 1: OpenTopoMap (excellent for Alpine topography)
        from cartopy.io.img_tiles import GoogleWTS
        
        class OpenTopoMap(GoogleWTS):
            def _image_url(self, tile):
                x, y, z = tile
                return f'https://tile.opentopomap.org/{z}/{x}/{y}.png'
        
        opentopomap = OpenTopoMap()
        if load_tiles_with_retry(ax, opentopomap, 14, max_retries=3):
            print("Successfully loaded OpenTopoMap tiles")
        else:
            raise Exception("OpenTopoMap failed after retries")

    except Exception as e:
        print(f"Could not load OpenTopoMap tiles ({e}), trying Google...")

        try:
            # Fallback 1: Google terrain
            google_terrain = cimgt.GoogleTiles(style='terrain')
            if load_tiles_with_retry(ax, google_terrain, 13, max_retries=2):
                print("Successfully loaded Google terrain tiles")
            else:
                raise Exception("Google terrain failed after retries")

        except Exception as e2:
            print(f"Could not load Google terrain tiles ({e2}), trying OSM...")

            try:
                # Fallback 2: OpenStreetMap
                osm_tiles = cimgt.OSM()
                if load_tiles_with_retry(ax, osm_tiles, 12, max_retries=2):
                    print("Successfully loaded OpenStreetMap tiles")
                else:
                    raise Exception("OSM failed after retries")

            except Exception as e3:
                print(f"Could not load OSM tiles ({e3}), using offline features...")

                # Fallback 3: Offline cartopy features (always works)
                ax.add_feature(cfeature.LAND, color='wheat', alpha=0.8)
                ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='gray')
                ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black', linestyle='--')
                ax.add_feature(cfeature.RIVERS, alpha=0.7, linewidth=1.5, color='blue')
                ax.add_feature(cfeature.LAKES, alpha=0.7, facecolor='lightblue', edgecolor='blue')
                print("Using offline cartopy features")
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle=':', 
                     color='black', linewidth=1, crs=data_crs)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}
    
    # Convert radar coordinates to lat/lon
    ranges = radar.range['data'] #/ 1000.0  # Convert to km
    azimuths = radar.azimuth['data']
    print(f"Azimuths shape: {azimuths.shape}")
    print(f"Azimuths values: {azimuths}")

    # Add this line to extract the field data for the selected field
    field_data = radar.fields[field_name]['data']

    # Use original arrays for plotting
    azimuths_padded = azimuths
    field_data_padded = field_data
    if field_name != 'snr' and 'snr' in radar.fields:
        snr_data_padded = radar.fields['snr']['data']

    # Apply azimuth offset to align with north (specific to this lidar installation)
    azimuth_offset = 286.1  # degrees - specific calibration for this lidar mounting
    azimuths_corrected = azimuths_padded + azimuth_offset

    # Create mesh of range and azimuth
    range_2d, azimuth_2d = np.meshgrid(ranges, azimuths_corrected)

    # Convert to Cartesian coordinates (x=east, y=north) in km
    x_km = range_2d * np.sin(np.radians(azimuth_2d))
    y_km = range_2d * np.cos(np.radians(azimuth_2d))

    # Convert to lat/lon
    delta_lat = y_km / 111.0
    delta_lon = x_km / (111.0 * np.cos(np.radians(LIDAR_LAT)))

    lats = LIDAR_LAT + delta_lat
    lons = LIDAR_LON + delta_lon

    # Apply SNR masking ONLY to velocity and beta fields, NOT to SNR itself
    if field_name != 'snr' and 'snr' in radar.fields:
        snr_mask = snr_data_padded < -19.5
        field_data_padded = np.ma.masked_where(snr_mask, field_data_padded)
        print(f"Applied SNR masking: {np.sum(snr_mask)} points masked out of {snr_mask.size} total")

    # Set up colormap and normalization based on field
    if field_name == 'velocity':
        cmap = cmocean.cm.balance
        vmin, vmax = -15, 15
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Radial Velocity (m/s)'
        title = 'Doppler Velocity'
    elif field_name == 'beta':
        cmap = 'plasma'
        norm = colors.LogNorm(vmin=1e-7, vmax=1e-4)
        cbar_label = 'Backscatter (m⁻¹ sr⁻¹)'
        title = 'Attenuated Backscatter'
    elif field_name == 'snr':
        # For SNR plots, show the full range without masking
        cmap = 'viridis'
        vmin, vmax = -20, 20  # Show full SNR range
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'SNR (dB)'
        title = 'Signal-to-Noise Ratio'
    else:
        cmap = 'viridis'
        vmin, vmax = 0, 1  # Set sensible defaults for unknown fields
        norm = None
        cbar_label = field_name
        title = field_name
    
    # Add elevation information to title if available
    if elevation is not None:
        elevation_str = f" ({elevation:.1f}° Elevation)"
    else:
        elevation_str = ""
        if hasattr(radar, "elevation") and hasattr(radar.elevation, "data"):
            try:
                elev_val = float(np.median(radar.elevation['data']))
                elevation_str = f" ({elev_val:.1f}° Elevation)"
            except Exception:
                pass

    title = f"{title}{elevation_str}"
    
    # Plot the data
    if field_name == 'snr':
        alpha_val = 0.5
    elif field_name == 'velocity':
        alpha_val = 0.8
    elif field_name == 'beta':
        alpha_val = 0.7
    else:
        alpha_val = 0.8
    
    # ...use field_data_padded for plotting...
    mesh = ax.pcolormesh(lons, lats, field_data_padded, 
                     transform=data_crs, cmap=cmap, norm=norm,
                     alpha=alpha_val, shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=40)
    cbar.set_label(cbar_label, fontsize=12, color='black', weight='bold')
    cbar.ax.tick_params(colors='black')
    
    # Mark lidar location
    ax.plot(LIDAR_LON, LIDAR_LAT, 'r*', markersize=20, 
            transform=data_crs, label='Lidar Location', zorder=10,
            markeredgecolor='black', markeredgewidth=2)
    
    # Add range rings
    range_rings = [0.5, 1.0, 1.5, 2.0]  # km
    theta = np.linspace(0, 2*np.pi, 100)
    
    for ring_range in range_rings:
        if ring_range <= max_range_km:
            x_ring = ring_range * np.cos(theta) / (111.0 * np.cos(np.radians(LIDAR_LAT)))
            y_ring = ring_range * np.sin(theta) / 111.0
            ax.plot(LIDAR_LON + x_ring, LIDAR_LAT + y_ring, 
                   'black', alpha=0.8, linewidth=2, transform=data_crs, linestyle='--')
            
            # Add range labels
            if ring_range < max_range_km:
                ax.text(LIDAR_LON + x_ring[25], LIDAR_LAT + y_ring[25], 
                       f'{ring_range:.1f}km', fontsize=10, 
                       transform=data_crs, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       color='black')
    
    # Set title
    ax.set_title(f'NCAS Doppler Lidar 1 - {title}\nSterzing, Italy - {timestamp_str}', 
                fontsize=14, pad=20, color='black', weight='bold')
    
    # Add legend
    legend = ax.legend(loc='upper right', facecolor='white', edgecolor='black')
    legend.get_texts()[0].set_color('black')
    
    # Add logos
    add_logos(fig, 0.5)
    
    return fig, ax

def create_ppi_map_plot_hybrid(radar, field_name='velocity', max_range_km=2.0, timestamp_str="", elevation=None):
    """
    Create a PPI map plot using Py-ART's RadarMapDisplay with Google terrain tiles as background.
    """
    fig = plt.figure(figsize=(12, 10))
    proj = ccrs.epsg(3857)  # Web Mercator
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    display = pyart.graph.RadarMapDisplay(radar)

    # Set up colormap and normalization based on field
    if field_name == 'velocity':
        cmap = cmocean.cm.balance
        vmin, vmax = -15, 15
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Radial Velocity (m/s)'
        title = 'Doppler Velocity'
    elif field_name == 'beta':
        cmap = 'plasma'
        norm = colors.LogNorm(vmin=1e-7, vmax=1e-4)
        cbar_label = 'Backscatter (m⁻¹ sr⁻¹)'
        title = 'Attenuated Backscatter'
    elif field_name == 'snr':
        cmap = 'viridis'
        vmin, vmax = -20, 20
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'SNR (dB)'
        title = 'Signal-to-Noise Ratio'
    else:
        cmap = 'viridis'
        vmin, vmax = 0, 1  # Set sensible defaults for unknown fields
        norm = None
        cbar_label = field_name
        title = field_name

    # Add elevation information to title if available
    if elevation is not None:
        elevation_str = f" ({elevation:.1f}° Elevation)"
    else:
        elevation_str = ""
        if hasattr(radar, "elevation") and hasattr(radar.elevation, "data"):
            try:
                elev_val = float(np.median(radar.elevation['data']))
                elevation_str = f" ({elev_val:.1f}° Elevation)"
            except Exception:
                pass

    title = f"{title}{elevation_str}"

    # Plot radar data using Py-ART RadarMapDisplay
    display = pyart.graph.RadarMapDisplay(radar)
    display.plot_ppi_map(
        field_name,
        sweep=0,
        ax=ax,
        min_lon=LIDAR_LON - max_range_km / 111.0,
        max_lon=LIDAR_LON + max_range_km / 111.0,
        min_lat=LIDAR_LAT - max_range_km / 111.0,
        max_lat=LIDAR_LAT + max_range_km / 111.0,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        colorbar_label=cbar_label,
        title=f'NCAS Doppler Lidar 1 - {title}\nSterzing, Italy - {timestamp_str}',
        projection=proj
    )

    # Set extent in Web Mercator for contextily
    lat_extent = max_range_km / 111.0
    lon_extent = max_range_km / (111.0 * np.cos(np.radians(LIDAR_LAT)))
    min_lon = LIDAR_LON - lon_extent
    max_lon = LIDAR_LON + lon_extent
    min_lat = LIDAR_LAT - lat_extent
    max_lat = LIDAR_LAT + lat_extent

    proj_wm = pyproj.Proj('epsg:3857')
    x0, y0 = proj_wm(min_lon, min_lat)
    x1, y1 = proj_wm(max_lon, max_lat)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Add Contextily OpenTopoMap basemap
    ctx.add_basemap(ax, crs='epsg:3857', source=ctx.providers.OpenTopoMap)

    # Mark lidar location
    lidar_x, lidar_y = proj_wm(LIDAR_LON, LIDAR_LAT)
    ax.plot(lidar_x, lidar_y, 'r*', markersize=20, label='Lidar Location', zorder=10,
            markeredgecolor='black', markeredgewidth=2)

    add_logos(fig, 0.5)
    return fig, ax

def create_ppi_map_plot_offline(
    radar, field_name='velocity', max_range_km=2.0, timestamp_str="", cache_dir=None, elevation=None
):
    """
    Create a PPI map plot using Py-ART's RadarMapDisplay and Contextily OpenTopoMap basemap.
    """
    import pyproj

    # Drop the last ray from azimuth, elevation, time, and all fields
    #for key in radar.fields:
    #    radar.fields[key]['data'] = radar.fields[key]['data'][:-1, :]
    #radar.azimuth['data'] = radar.azimuth['data'][:-1]
    #radar.elevation['data'] = radar.elevation['data'][:-1]
    #radar.time['data'] = radar.time['data'][:-1]

    fig = plt.figure(figsize=(12, 10))
    proj = ccrs.epsg(3857)  # Web Mercator
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Set up colormap and normalization based on field
    if field_name == 'velocity':
        cmap = cmocean.cm.balance
        vmin, vmax = -15, 15
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Radial Velocity (m/s)'
        title = 'Doppler Velocity'
    elif field_name == 'beta':
        cmap = 'plasma'
        vmin, vmax = 1e-7, 1e-4
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        cbar_label = 'Backscatter (m⁻¹ sr⁻¹)'
        title = 'Attenuated Backscatter'
    elif field_name == 'snr':
        cmap = 'viridis'
        vmin, vmax = -20, 20
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'SNR (dB)'
        title = 'Signal-to-Noise Ratio'
    else:
        cmap = 'viridis'
        vmin, vmax = 0, 1
        norm = None
        cbar_label = field_name
        title = field_name

    # Add elevation information to title if available
    if elevation is not None:
        elevation_str = f" ({elevation:.1f}° Elevation)"
    else:
        elevation_str = ""
        if hasattr(radar, "elevation") and hasattr(radar.elevation, "data"):
            try:
                elev_val = float(np.median(radar.elevation['data']))
                elevation_str = f" ({elev_val:.1f}° Elevation)"
            except Exception:
                pass

    title = f"{title}{elevation_str}"

    print(f"radar['range']: {radar.range['data']}")
    print(f"radar['azimuth']: {radar.azimuth['data']}")
    print(f"radar['elevation']: {radar.elevation['data']}")

    # Set extent in Web Mercator for contextily
    lat_extent = max_range_km / 111.0
    lon_extent = max_range_km / (111.0 * np.cos(np.radians(LIDAR_LAT)))
    min_lon = LIDAR_LON - lon_extent
    max_lon = LIDAR_LON + lon_extent
    min_lat = LIDAR_LAT - lat_extent
    max_lat = LIDAR_LAT + lat_extent

    print(f"Map extent: {min_lon:.4f} to {max_lon:.4f}, {min_lat:.4f} to {max_lat:.4f}")

    # Plot radar data using Py-ART RadarMapDisplay
    display = pyart.graph.RadarMapDisplay(radar)
    
    # Create a gate filter for the radar object
    gatefilter = pyart.filters.GateFilter(radar)
    if 'snr' in radar.fields:
        gatefilter.exclude_below('snr', -19.5)  # Mask gates where SNR < -19.5 dB

    # Pass the gatefilter to plot_ppi_map
    display.plot_ppi_map(
        field_name,
        sweep=0,
        ax=ax,
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        colorbar_label=cbar_label,
        title=f'NCAS Doppler Lidar 1 - {title}\nSterzing, Italy - {timestamp_str}',
        projection=proj,
        gatefilter=gatefilter  # <-- Apply the gate filter here
    )

    
    proj_wm = pyproj.Proj('epsg:3857')
    x0, y0 = proj_wm(min_lon, min_lat)
    x1, y1 = proj_wm(max_lon, max_lat)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
 
    # Add Contextily OpenTopoMap basemap
    ctx.add_basemap(ax, crs='epsg:3857', source=ctx.providers.OpenTopoMap)

    # After display.plot_ppi_map(...)
    gl = ax.gridlines(
    draw_labels=True,
    linewidth=1,
    color='gray',
    alpha=0.5,
    linestyle='--',
    crs=ccrs.PlateCarree()
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}

    # Mark lidar location
    #lidar_x, lidar_y = proj_wm(LIDAR_LON, LIDAR_LAT)
    #ax.plot(lidar_x, lidar_y, 'r*', markersize=20, label='Lidar Location', zorder=10,
    #        markeredgecolor='black', markeredgewidth=2)

    add_logos(fig, 0.5)
    return fig, ax

def create_ppi_map_plot_offline_old(radar, field_name='velocity', max_range_km=2.0, timestamp_str="", cache_dir=None, elevation=None):
    """
    Create a PPI map plot with offline-first tile loading
    """
    
    # Calculate approximate extent in degrees
    lat_extent = max_range_km / 111.0
    lon_extent = max_range_km / (111.0 * np.cos(np.radians(LIDAR_LAT)))
    
    # Use Web Mercator projection (native for most tile services)
    proj = ccrs.epsg(3857)  # Web Mercator
    data_crs = ccrs.PlateCarree()  # For our lat/lon data
    
    # Create figure and axis
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=proj)
    
    # Transform extent to Web Mercator
    extent_lonlat = [LIDAR_LON - lon_extent, LIDAR_LON + lon_extent,
                     LIDAR_LAT - lat_extent, LIDAR_LAT + lat_extent]
    
    ax.set_extent(extent_lonlat, crs=data_crs)
    
    # Set default cache directory if not provided
    if cache_dir is None:
        cache_dir = "/home/users/cjwalden/git/halo-teamx/tile_cache"
    
    # Try to use cached tiles, fall back to online if not available
    try:
        # Try to use cached tiles
        if os.path.exists(cache_dir):
            print("Using cached OpenTopoMap tiles...")
            offline_tiles = OfflineOpenTopoMap(cache_dir=cache_dir)
            if load_tiles_with_retry(ax, offline_tiles, 14, max_retries=1):
                print("Successfully loaded cached tiles")
            else:
                raise Exception("Cached tiles failed")
        else:
            raise Exception("No tile cache found")
            
    except Exception as e:
        print(f"Cached tiles failed ({e}), trying online OpenTopoMap...")
        
        try:
            # Fallback to online OpenTopoMap
            class OpenTopoMap(cimgt.GoogleWTS):
                def _image_url(self, tile):
                    x, y, z = tile
                    return f'https://tile.opentopomap.org/{z}/{x}/{y}.png'
            
            opentopomap = OpenTopoMap()
            if load_tiles_with_retry(ax, opentopomap, 14, max_retries=3):
                print("Successfully loaded online OpenTopoMap tiles")
            else:
                raise Exception("Online OpenTopoMap failed")

        except Exception as e2:
            print(f"OpenTopoMap failed ({e2}), trying Google...")

            try:
                # Fallback to Google terrain
                google_terrain = cimgt.GoogleTiles(style='terrain')
                if load_tiles_with_retry(ax, google_terrain, 13, max_retries=2):
                    print("Successfully loaded Google terrain tiles")
                else:
                    raise Exception("Google terrain failed")

            except Exception as e3:
                print(f"Google failed ({e3}), trying OSM...")

                try:
                    # Fallback to OpenStreetMap
                    osm_tiles = cimgt.OSM()
                    if load_tiles_with_retry(ax, osm_tiles, 12, max_retries=2):
                        print("Successfully loaded OpenStreetMap tiles")
                    else:
                        raise Exception("OSM failed")

                except Exception as e4:
                    print(f"All online tiles failed ({e4}), using offline features...")

                    # Final fallback: Offline cartopy features
                    ax.add_feature(cfeature.LAND, color='wheat', alpha=0.8)
                    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='gray')
                    ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black', linestyle='--')
                    ax.add_feature(cfeature.RIVERS, alpha=0.7, linewidth=1.5, color='blue')
                    ax.add_feature(cfeature.LAKES, alpha=0.7, facecolor='lightblue', edgecolor='blue')
                    print("Using offline cartopy features")
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle=':', 
                     color='black', linewidth=1, crs=data_crs)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 10, 'color': 'black', 'weight': 'bold'}
    
    # Convert radar coordinates to lat/lon
    ranges = radar.range['data'] #/ 1000.0  # Convert to km
    azimuths = radar.azimuth['data']

    # Add this line to extract the field data for the selected field
    field_data = radar.fields[field_name]['data']

    # Use original arrays for plotting
    azimuths_padded = azimuths
    field_data_padded = field_data
    if field_name != 'snr' and 'snr' in radar.fields:
        snr_data_padded = radar.fields['snr']['data']

    # Apply azimuth offset to align with north (specific to this lidar installation)
    azimuth_offset = 286.1  # degrees - specific calibration for this lidar mounting
    azimuths_corrected = azimuths_padded + azimuth_offset

    print(f"Azimuths shape: {azimuths_corrected.shape}")
    print(f"Azimuths values: {azimuths_corrected}")

    # Create mesh of range and azimuth
    range_2d, azimuth_2d = np.meshgrid(ranges, azimuths_corrected)

    # Convert to Cartesian coordinates (x=east, y=north) in km
    x_km = range_2d * np.sin(np.radians(azimuth_2d))
    y_km = range_2d * np.cos(np.radians(azimuth_2d))

    # Convert to lat/lon
    delta_lat = y_km / 111.0
    delta_lon = x_km / (111.0 * np.cos(np.radians(LIDAR_LAT)))

    lats = LIDAR_LAT + delta_lat
    lons = LIDAR_LON + delta_lon

    # Apply SNR masking ONLY to velocity and beta fields, NOT to SNR itself
    if field_name != 'snr' and 'snr' in radar.fields:
        snr_mask = snr_data_padded < -19.5
        field_data_padded = np.ma.masked_where(snr_mask, field_data_padded)
        print(f"Applied SNR masking: {np.sum(snr_mask)} points masked out of {snr_mask.size} total")

    # Set up colormap and normalization based on field
    if field_name == 'velocity':
        cmap = cmocean.cm.balance
        vmin, vmax = -15, 15
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Radial Velocity (m/s)'
        title = 'Doppler Velocity'
    elif field_name == 'beta':
        cmap = 'plasma'
        vmin = 1e-7
        vmax = 1e-4
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        cbar_label = 'Backscatter (m⁻¹ sr⁻¹)'
        title = 'Attenuated Backscatter'
    elif field_name == 'snr':
        # For SNR plots, show the full range without masking
        cmap = 'viridis'
        vmin, vmax = -20, 20  # Show full SNR range
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'SNR (dB)'
        title = 'Signal-to-Noise Ratio'
    else:
        cmap = 'viridis'
        vmin, vmax = 0, 1  # Set sensible defaults for unknown fields
        norm = None
        cbar_label = field_name
        title = field_name
    
    # Add elevation information to title if available
    if elevation is not None:
        elevation_str = f" ({elevation:.1f}° Elevation)"
    else:
        elevation_str = ""
        if hasattr(radar, "elevation") and hasattr(radar.elevation, "data"):
            try:
                elev_val = float(np.median(radar.elevation['data']))
                elevation_str = f" ({elev_val:.1f}° Elevation)"
            except Exception:
                pass

    title = f"{title}{elevation_str}"
    
    # Plot the data
    if field_name == 'snr':
        alpha_val = 0.5
    elif field_name == 'velocity':
        alpha_val = 0.8
    elif field_name == 'beta':
        alpha_val = 0.7
    else:
        alpha_val = 0.8
    
    # ...use field_data_padded for plotting...
    mesh = ax.pcolormesh(lons, lats, field_data_padded, 
                     transform=data_crs, cmap=cmap, norm=norm,
                     alpha=alpha_val, shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=40)
    cbar.set_label(cbar_label, fontsize=12, color='black', weight='bold')
    cbar.ax.tick_params(colors='black')
    
    # Mark lidar location
    ax.plot(LIDAR_LON, LIDAR_LAT, 'r*', markersize=20, 
            transform=data_crs, label='Lidar Location', zorder=10,
            markeredgecolor='black', markeredgewidth=2)
    
    # Add range rings
    range_rings = [0.5, 1.0, 1.5, 2.0]  # km
    theta = np.linspace(0, 2*np.pi, 100)
    
    for ring_range in range_rings:
        if ring_range <= max_range_km:
            x_ring = ring_range * np.cos(theta) / (111.0 * np.cos(np.radians(LIDAR_LAT)))
            y_ring = ring_range * np.sin(theta) / 111.0
            ax.plot(LIDAR_LON + x_ring, LIDAR_LAT + y_ring, 
                   'black', alpha=0.8, linewidth=2, transform=data_crs, linestyle='--')
            
            # Add range labels
            if ring_range < max_range_km:
                ax.text(LIDAR_LON + x_ring[25], LIDAR_LAT + y_ring[25], 
                       f'{ring_range:.1f}km', fontsize=10, 
                       transform=data_crs, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       color='black')
    
    # Set title
    ax.set_title(f'NCAS Doppler Lidar 1 - {title}\nSterzing, Italy - {timestamp_str}', 
                fontsize=14, pad=20, color='black', weight='bold')
    
    # Add legend
    legend = ax.legend(loc='upper right', facecolor='white', edgecolor='black')
    legend.get_texts()[0].set_color('black')
    
    # Add logos
    add_logos(fig, 0.5)
    
    return fig, ax

def setup_tile_cache():
    """Setup and pre-download tiles for offline use"""
    cache_dir = Path("/home/users/cjwalden/git/halo-teamx/tile_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate area around lidar (2km radius)
    lat_extent = 2.0 / 111.0  # 2km in degrees
    lon_extent = 2.0 / (111.0 * np.cos(np.radians(LIDAR_LAT)))
    
    lat_min = LIDAR_LAT - lat_extent
    lat_max = LIDAR_LAT + lat_extent
    lon_min = LIDAR_LON - lon_extent
    lon_max = LIDAR_LON + lon_extent
    
    print(f"Pre-downloading tiles for area: {lat_min:.4f} to {lat_max:.4f}, {lon_min:.4f} to {lon_max:.4f}")
    
    download_tile_cache(lat_min, lat_max, lon_min, lon_max, 
                       zoom_levels=[12, 13, 14], cache_dir=cache_dir)
    
    return cache_dir

def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile numbers"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def download_tile_cache(lat_min, lat_max, lon_min, lon_max, zoom_levels, cache_dir):
    """Download tiles for offline use"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # OpenTopoMap URL template
    url_template = "https://tile.opentopomap.org/{z}/{x}/{y}.png"
    
    total_tiles = 0
    downloaded_tiles = 0
    
    for zoom in zoom_levels:
        print(f"Downloading zoom level {zoom}...")
        
        # Calculate tile bounds
        x_min, y_max = deg2num(lat_min, lon_min, zoom)
        x_max, y_min = deg2num(lat_max, lon_max, zoom)
        
        zoom_dir = cache_dir / str(zoom)
        zoom_dir.mkdir(exist_ok=True)
        
        for x in range(x_min, x_max + 1):
            x_dir = zoom_dir / str(x)
            x_dir.mkdir(exist_ok=True)
            
            for y in range(y_min, y_max + 1):
                tile_file = x_dir / f"{y}.png"
                total_tiles += 1
                
                if not tile_file.exists():
                    try:
                        url = url_template.format(z=zoom, x=x, y=y)
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        
                        with open(tile_file, 'wb') as f:
                            f.write(response.content)
                        
                        downloaded_tiles += 1
                        if downloaded_tiles % 10 == 0:
                            print(f"  Downloaded {downloaded_tiles} tiles...")
                        
                        # Be nice to the server
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"  Failed to download tile {x}/{y}: {e}")
                else:
                    print(f"  Tile {x}/{y} already cached")
    
    print(f"Download complete: {downloaded_tiles} new tiles, {total_tiles} total")

class OfflineOpenTopoMap(cimgt.GoogleWTS):
    """OpenTopoMap tile source that uses cached tiles first"""
    
    def __init__(self, cache_dir=None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else None
    
    def _image_url(self, tile):
        x, y, z = tile
        
        # Check if tile exists locally first
        if self.cache_dir:
            tile_file = self.cache_dir / str(z) / str(x) / f"{y}.png"
            if tile_file.exists():
                return tile_file.as_uri()  # Return file:// URL
        
        # Fallback to online
        return f'https://tile.opentopomap.org/{z}/{x}/{y}.png'

def process_single_file(selected_file, file_index, total_files, quicklook_dir, map_style='standard', cache_dir=None, elevation=None):
    """
    Process a single VAD/PPI file and create map plots
    """
    
    # Set default cache directory
    if cache_dir is None:
        cache_dir = "/home/users/cjwalden/git/halo-teamx/tile_cache"
    
    try:
        # Load single .hpl file using doppy's raw data loader
        try:
            # Use HaloHpl.from_srcs method
            halo_data_list = doppy.raw.HaloHpl.from_srcs(
                [selected_file], 
                overlapped_gates=True
            )
            
            # Get the first (and only) data object
            lidar_data = halo_data_list[0]
            
        except Exception as e:
            print(f"Error loading HPL file {selected_file}: {e}")
            return []
        
        # Check scan type from the header - only process VAD or PPI scans
        try:
            # Check if the lidar_data has scan_type or header information
            scan_type = None
            
            # Try different ways to access scan type information
            if hasattr(lidar_data, 'scan_type'):
                scan_type = str(lidar_data.scan_type).upper()
            elif hasattr(lidar_data, 'header') and hasattr(lidar_data.header, 'scan_type'):
                scan_type = str(lidar_data.header.scan_type).upper()
            elif hasattr(lidar_data, 'attrs') and 'scan_type' in lidar_data.attrs:
                scan_type = str(lidar_data.attrs['scan_type']).upper()
            else:
                # If we can't find scan_type, check the filename or try to read header directly
                filename = os.path.basename(selected_file)
                
                # Try to read the header directly from the file
                try:
                    with open(selected_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f):
                            if line_num > 50:  # Don't read too far into the file
                                break
                            if 'scan type:' in line.lower() or 'scan_type' in line.lower():
                                scan_type = line.split(':')[-1].strip().upper()
                                break
                except Exception:
                    pass
            
            # Check if this is a VAD or PPI scan
            if scan_type is None:
                print(f"Warning: Could not determine scan type for {selected_file}")
                # Continue processing anyway, but with a warning
            elif 'VAD' not in scan_type and 'PPI' not in scan_type:
                print(f"Skipping {selected_file}: Scan type '{scan_type}' is not VAD or PPI")
                return []
            else:
                print(f"Processing {selected_file}: Scan type '{scan_type}'")
                
        except Exception as e:
            print(f"Warning: Error checking scan type for {selected_file}: {e}")
        
        # Create PyART radar object manually
        try:
            # Extract data from lidar_data object using correct attribute names
            if hasattr(lidar_data.radial_distance, 'values'):
                range_m = lidar_data.radial_distance.values
            else:
                range_m = lidar_data.radial_distance
                
            if hasattr(lidar_data.azimuth, 'values'):
                azimuth_data = lidar_data.azimuth.values
            else:
                azimuth_data = lidar_data.azimuth
            
            print(f"DEBUG - Range shape: {range_m.shape}, Azimuth shape: {azimuth_data.shape}")
            
            # Apply azimuth offset to align with north (specific to this lidar installation)
            azimuth_offset = 286.1  # degrees - specific calibration for this lidar mounting
            azimuth_data = azimuth_data + azimuth_offset

            # Now create the radar object using the offset azimuths
            radar = pyart.testing.make_empty_ppi_radar(
                ngates=len(range_m),
                rays_per_sweep=len(azimuth_data),
                nsweeps=1
            )
            
            # Update radar object with our data
            radar.range['data'] = range_m.astype(np.float32)
            radar.azimuth['data'] = azimuth_data.astype(np.float32)
            
            # Extract the actual elevation for this scan
            if hasattr(lidar_data, 'elevation'):
                if hasattr(lidar_data.elevation, 'values'):
                    elev_val = float(lidar_data.elevation.values[0])
                else:
                    elev_val = float(lidar_data.elevation[0])
            elif hasattr(lidar_data, 'phi'):
                if hasattr(lidar_data.phi, 'values'):
                    elev_val = float(lidar_data.phi.values[0])
                else:
                    elev_val = float(lidar_data.phi[0])
            else:
                elev_val = 5.0  # fallback

            # Then use this value for the radar object:
            radar.elevation['data'] = np.full_like(azimuth_data, elev_val, dtype=np.float32)
            radar.time['data'] = np.arange(len(azimuth_data), dtype=np.float32)
            
            # Add available fields to radar object
            fields = {}
            
            print(f"DEBUG - Converting fields to radar object:")
            
            # Check for velocity field - try multiple possible names
            velocity_found = False
            for vel_name in ['velocity', 'radial_velocity', 'doppler_velocity']:
                if hasattr(lidar_data, vel_name):
                    vel_attr = getattr(lidar_data, vel_name)
                    if vel_attr is not None:
                        try:
                            print(f"DEBUG: Found velocity data under attribute '{vel_name}'")
                            if hasattr(vel_attr, 'values'):
                                velocity_data = vel_attr.values
                            else:
                                velocity_data = vel_attr
                            
                            # Handle different dimensionalities
                            if len(velocity_data.shape) == 2:
                                vel_data = velocity_data  # Data is [azimuth, range]
                            elif len(velocity_data.shape) == 3:
                                vel_data = velocity_data[0, :, :]  # Use first time step
                            else:
                                print(f"DEBUG: Unexpected velocity shape: {velocity_data.shape}")
                                continue
                            
                            print(f"DEBUG: Velocity data shape: {vel_data.shape}, dtype: {vel_data.dtype}")
                            print(f"DEBUG: Velocity data range: {np.nanmin(vel_data):.3f} to {np.nanmax(vel_data):.3f}")
                            
                            fields['velocity'] = {
                                'data': vel_data.astype(np.float32),
                                'units': 'm/s',
                                'long_name': 'Radial velocity',
                                '_FillValue': np.nan
                            }
                            print(f"DEBUG: Successfully added velocity field to radar object")
                            velocity_found = True
                            break
                        except Exception as e:
                            print(f"DEBUG: Error processing velocity field '{vel_name}': {e}")
            
            if not velocity_found:
                print(f"DEBUG: No velocity field successfully converted")
            
            # Check for backscatter field - try multiple possible names
            beta_found = False
            for beta_name in ['beta', 'backscatter', 'attenuated_backscatter']:
                if hasattr(lidar_data, beta_name):
                    beta_attr = getattr(lidar_data, beta_name)
                    if beta_attr is not None:
                        try:
                            print(f"DEBUG: Found beta data under attribute '{beta_name}'")
                            if hasattr(beta_attr, 'values'):
                                beta_data = beta_attr.values
                            else:
                                beta_data = beta_attr
                            
                            # Handle different dimensionalities
                            if len(beta_data.shape) == 2:
                                beta_data_2d = beta_data  # Data is [azimuth, range]
                            elif len(beta_data.shape) == 3:
                                beta_data_2d = beta_data[0, :, :]  # Use first time step
                            else:
                                print(f"DEBUG: Unexpected beta shape: {beta_data.shape}")
                                continue
                            
                            print(f"DEBUG: Beta data shape: {beta_data_2d.shape}, dtype: {beta_data_2d.dtype}")
                            print(f"DEBUG: Beta data range: {np.nanmin(beta_data_2d):.6e} to {np.nanmax(beta_data_2d):.6e}")
                            
                            fields['beta'] = {
                                'data': beta_data_2d.astype(np.float32),
                                'units': 'm-1 sr-1',
                                'long_name': 'Attenuated backscatter coefficient',
                                '_FillValue': np.nan
                            }
                            print(f"DEBUG: Successfully added beta field to radar object")
                            beta_found = True
                            break
                        except Exception as e:
                            print(f"DEBUG: Error processing beta field '{beta_name}': {e}")
            
            if not beta_found:
                print(f"DEBUG: No beta field successfully converted")
            
            # Calculate SNR from intensity (like in quicklook_ppi_halo.py)
            snr_found = False
            if hasattr(lidar_data, 'intensity'):
                intensity_attr = getattr(lidar_data, 'intensity')
                if intensity_attr is not None:
                    try:
                        print(f"DEBUG: Found intensity data, calculating SNR")
                        if hasattr(intensity_attr, 'values'):
                            intensity_data = intensity_attr.values
                        else:
                            intensity_data = intensity_attr
                        
                        # Handle different dimensionalities
                        if len(intensity_data.shape) == 2:
                            int_data = intensity_data  # Data is [azimuth, range]
                        elif len(intensity_data.shape) == 3:
                            int_data = intensity_data[0, :, :]  # Use first time step
                        else:
                            print(f"DEBUG: Unexpected intensity shape: {intensity_data.shape}")
                            int_data = None
                        
                        if int_data is not None:
                            # Calculate SNR from intensity (like in quicklook_ppi_halo.py)
                            snr_linear = int_data - 1  # Convert intensity to SNR (linear)
                            snr_db = 10 * np.log10(np.maximum(snr_linear, 0.01))  # Convert to dB
                            
                            print(f"DEBUG: SNR data shape: {snr_db.shape}, dtype: {snr_db.dtype}")
                            print(f"DEBUG: SNR data range: {np.nanmin(snr_db):.3f} to {np.nanmax(snr_db):.3f} dB")
                            print(f"DEBUG: SNR valid points: {np.sum(~np.isnan(snr_db))}/{snr_db.size}")
                            
                            fields['snr'] = {
                                'data': snr_db.astype(np.float32),
                                'units': 'dB',
                                'long_name': 'Signal-to-noise ratio',
                                '_FillValue': np.nan
                            }
                            print(f"DEBUG: Successfully calculated and added SNR field to radar object")
                            snr_found = True
                        
                    except Exception as e:
                        print(f"DEBUG: Error calculating SNR from intensity: {e}")
                        import traceback
                        traceback.print_exc()
            
            if not snr_found:
                print(f"DEBUG: No SNR field successfully calculated from intensity")
            
            # Add fields to radar object
            radar.fields = fields
            print(f"DEBUG: Final radar object has fields: {list(radar.fields.keys())}")
            
            # After creating the radar object, set its latitude and longitude using LIDAR_LAT and LIDAR_LON
            radar.latitude['data'] = np.array([LIDAR_LAT], dtype=np.float32)
            radar.longitude['data'] = np.array([LIDAR_LON], dtype=np.float32)
        
        except Exception as e:
            print(f"Error creating radar object from {selected_file}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Get timestamp from filename for plot titles
        filename = os.path.basename(selected_file)
        try:
            timestamp_part = filename.split('_')[3].split('.')[0]  # HHMMSS
            date_part = filename.split('_')[2]  # YYYYMMDD
            timestamp_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {timestamp_part[:2]}:{timestamp_part[2:4]}:{timestamp_part[4:6]} UTC"
        except:
            timestamp_str = filename
        
        # Available fields to plot
        available_fields = []
        
        if 'velocity' in radar.fields:
            available_fields.append(('velocity', 'Doppler Velocity', 'Radial Velocity (m/s)'))
            print(f"DEBUG: Added velocity to available_fields")
        
        if 'beta' in radar.fields:
            available_fields.append(('beta', 'Attenuated Backscatter', 'Backscatter (m⁻¹ sr⁻¹)'))
            print(f"DEBUG: Added beta to available_fields")
        
        if 'snr' in radar.fields:
            available_fields.append(('snr', 'Signal-to-Noise Ratio', 'SNR (dB)'))
            print(f"DEBUG: Added snr to available_fields")
        
        print(f"DEBUG: Final available_fields list: {available_fields}")
        
        if not available_fields:
            print(f"No valid fields found in {filename}")
            return []
        
        output_files = []
        
        # Create plots for each available field
        print(f"DEBUG: Starting to create plots for {len(available_fields)} fields")
        for field_index, (field_name, field_title, field_units) in enumerate(available_fields):
            print(f"DEBUG: Processing field {field_index+1}/{len(available_fields)}: {field_name}")
            try:
                # Choose plotting function based on map style
                if map_style == 'hybrid':
                    print(f"DEBUG: Creating hybrid plot for {field_name}")
                    fig, ax = create_ppi_map_plot_hybrid(radar, field_name=field_name, 
                                                       max_range_km=2.0, timestamp_str=timestamp_str)
                else:
                    print(f"DEBUG: Creating standard plot for {field_name}")
                    fig, ax = create_ppi_map_plot_offline(radar, field_name=field_name, 
                                                        max_range_km=2.0, timestamp_str=timestamp_str,
                                                        cache_dir=cache_dir, elevation=elevation)
                
                # Generate output filename with new naming convention
                try:
                    timestamp_part = filename.split('_')[3].split('.')[0]  # HHMMSS
                    date_part = filename.split('_')[2]  # YYYYMMDD
                    datetime_str = f"{date_part}-{timestamp_part[:2]}{timestamp_part[2:4]}{timestamp_part[4:6]}"
                    output_filename = (
                        f"ncas-lidar-dop-1_sterzing_{field_name}_ppi-map_{map_style}_{datetime_str}.png"
                )
                except Exception:
                    output_filename = f"ncas-lidar-dop-1_sterzing_{field_name}_ppi-map_{map_style}_{filename}.png"
        
                output_filepath = quicklook_dir / output_filename

                # Save the plot
                print(f"DEBUG: Saving plot to {output_filepath}")
                plt.savefig(output_filepath, dpi=150, bbox_inches='tight', 
                            facecolor='white', edgecolor='none')
                plt.close(fig)  # Important: close figure to free memory

                output_files.append(output_filename)
                print(f"DEBUG: Successfully created and saved {field_name} plot")

            except Exception as e:
                print(f"ERROR: Failed to create {field_name} plot for {filename}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next field instead of breaking
                continue

        print(f"DEBUG: Completed processing. Created {len(output_files)} plots: {output_files}")
        return output_files
        
    except Exception as e:
        print(f"Error processing file {selected_file}: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create PPI map plots for HALO lidar data')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--max_files', default=None, type=int, help='Maximum number of files to process')
    parser.add_argument('--map_style', default='standard', choices=['standard', 'hybrid'], 
                       help='Map style: standard or hybrid')
    parser.add_argument('--download_tiles', action='store_true', 
                       help='Download tiles for offline use before processing')
    parser.add_argument('--elevation', type=float, default=None, help='Only process scans with this elevation (deg)')
    args = parser.parse_args()
    
    # Set up tile cache
    cache_dir = "/home/users/cjwalden/git/halo-teamx/tile_cache"
    
    if args.download_tiles:
        print("Pre-downloading tiles for offline use...")
        cache_dir = setup_tile_cache()
        print(f"Tile cache ready at: {cache_dir}")
    
    datestr = args.date
    max_files = args.max_files
    map_style = args.map_style
    
    # Extract year and month
    yr = datestr[0:4]
    mo = datestr[0:6]
    
    # Set up paths
    teamx_halo_path = "/gws/pw/j07/ncas_obs_vol1/amf/raw_data/ncas-lidar-dop-1/incoming/20250603_teamx/Proc"
    datepath = os.path.join(teamx_halo_path, yr, mo, datestr)
    
    # Set up output directory
    base_out_dir = Path("/gws/pw/j07/ncas_obs_vol1/amf/processing/ncas-lidar-dop-1/20250603_teamx/doppy_processed")
    quicklook_dir = base_out_dir / "quicklooks" / "ppi_maps" / datestr  
    quicklook_dir.mkdir(parents=True, exist_ok=True)
    
    # Find VAD/PPI files
    os.chdir(datepath)
    VAD_files = [os.path.join(datepath, f) for f in glob.glob(f'User5_18_{datestr}_*.hpl')]

    # Filter by elevation if requested
    if args.elevation is not None:
        filtered_files = []
        for f in VAD_files:
            try:
                halo_data_list = doppy.raw.HaloHpl.from_srcs([f], overlapped_gates=True)
                lidar_data = halo_data_list[0]
                elev = None
                if hasattr(lidar_data, 'elevation'):
                    if hasattr(lidar_data.elevation, 'values'):
                       
                        elev = float(lidar_data.elevation.values[0])
                    else:
                        elev = float(lidar_data.elevation[0])
                
                print(f"DEBUG: File {f} - Elevation: {elev}")
                
                if elev is not None and abs(elev - args.elevation) < 0.1:
                    filtered_files.append(f)
                    print(f"DEBUG: Added {f} to filtered files")
                else:
                    print(f"DEBUG: Skipped {f}, elevation {elev} does not match")
            
            except Exception as e:
                print(f"ERROR: Failed to process file {f} for elevation filtering: {e}")
                import traceback
                traceback.print_exc()
        
        VAD_files = filtered_files
        print(f"DEBUG: Total files after elevation filtering: {len(VAD_files)}")
    
    # Limit number of files to process
    if max_files is not None and len(VAD_files) > max_files:
        VAD_files = VAD_files[:max_files]
        print(f"INFO: Limiting to first {max_files} files")
    
    # Process each file
    all_output_files = []
    for file_index, selected_file in enumerate(VAD_files):
        print(f"INFO: Processing file {file_index+1}/{len(VAD_files)}: {selected_file}")
        output_files = process_single_file(selected_file, file_index, len(VAD_files), quicklook_dir, 
                                           map_style=map_style, cache_dir=cache_dir, elevation=args.elevation)
        all_output_files.extend(output_files)
    
    print(f"INFO: Completed processing {len(VAD_files)} files. Created output files: {all_output_files}")

if __name__ == "__main__":
    main()
