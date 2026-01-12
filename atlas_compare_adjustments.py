#!/usr/bin/env python3
"""
Atlas plotting script for deployment data comparison
Creates difference plots between .adj and .dft files for each instrument
Usage: python atlas_compare_adjustments.py <base_filename_without_extension>
"""

import pandas as pd
import sys
import os
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, DataRange1d, DatetimeTickFormatter
from bokeh.layouts import gridplot
from bokeh.io import export_png


def detect_parameter_type(filename):
    """Detect parameter type from filename"""
    basename = os.path.basename(filename).lower()

    # Determine parameter type
    if 'sal' in basename:
        param_type = 'salinity'
        diff_label = 'Salinity Difference (PSU)'
    elif 'cond' in basename:
        param_type = 'conductivity'
        diff_label = 'Conductivity Difference (mS/cm)'
    elif 'dens' in basename:
        param_type = 'density'
        diff_label = 'Density Difference (ρ₀)'
    else:
        param_type = 'unknown'
        diff_label = 'Difference'

    return param_type, diff_label


def parse_dft_file(file_path):
    """Parse .dft and .adj deployment files (identical format)"""
    file_ext = os.path.splitext(file_path)[1]
    print(f"Reading {file_ext.upper()} file: {file_path}")

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Skip first 3 lines, get headers from lines 4-5
        header_line2 = lines[4].strip()  # Depths

        # Extract depths from header line 2
        header2_parts = header_line2.split()
        depths = []
        for part in header2_parts:
            try:
                depth_val = int(part)
                if 0 < depth_val < 1000:  # Reasonable depth range
                    depths.append(str(depth_val))
            except ValueError:
                continue

        # Create column names for all depths
        columns = ['DATE']
        for depth in depths:
            if depth == '1':
                columns.append('SSS')
            else:
                columns.append(f'S{int(depth):03d}')

        # Parse data starting from line 6
        data_rows = []
        line_count = 0

        for line in lines[5:]:
            parts = line.strip().split()
            if len(parts) < len(depths) + 1:  # Need at least date + all depth values
                continue

            line_count += 1

            # Parse date (YYYYDDDHHMM format)
            date_str = parts[0]
            try:
                dt = pd.to_datetime(date_str, format='%Y%j%H%M%S')
            except:
                continue

            # Extract values for all depths
            try:
                values = []
                for i in range(1, len(depths) + 1):
                    if i < len(parts):
                        val = float(parts[i])
                        # Filter out bad values
                        if val > 1000 or val < -100:
                            val = None
                        values.append(val)
                    else:
                        values.append(None)

                # Create row: [date] + [val1, val2, val3, val4]
                row = [dt] + values
                data_rows.append(row)

            except Exception:
                continue

        # Create DataFrame
        if data_rows:
            df = pd.DataFrame(data_rows)
            df.columns = columns
            df.set_index('DATE', inplace=True)
        else:
            # Create empty DataFrame with proper structure
            df = pd.DataFrame()
            for col in columns[1:]:  # Skip DATE column
                df[col] = pd.Series(dtype=float)
            df.index = pd.DatetimeIndex([], name='DATE')

        print(f"Processed {line_count} data lines, kept {len(data_rows)} valid rows")
        print(f"Created DataFrame with shape: {df.shape}")

        return df

    except Exception as e:
        print(f"Error parsing {file_ext.upper()} file: {e}")
        return None


def create_depth_label(col):
    """Create appropriate label for depth column"""
    if col == 'SSS':
        return '1m (SSS)'
    elif col.startswith('S'):
        try:
            depth = int(col[1:])
            return f'{depth}m'
        except:
            return col
    else:
        return col


def get_hover_units(param_type):
    """Get appropriate units for hover tooltips"""
    if param_type == 'salinity':
        return 'PSU'
    elif param_type == 'conductivity':
        return 'mS/cm'
    elif param_type == 'density':
        return 'ρ₀'
    else:
        return ''


def create_difference_subplot(df_adj, df_dft, col, param_type, diff_label, shared_x_range):
    """Create subplot showing difference between .adj and .dft files"""

    # Align the dataframes by index (datetime)
    common_index = df_adj.index.intersection(df_dft.index)

    if len(common_index) == 0:
        print(f"No common timestamps for column {col}")
        return None

    # Calculate differences (adj - dft)
    adj_data = df_adj.loc[common_index, col]
    dft_data = df_dft.loc[common_index, col]

    # Only calculate differences where both values are not NaN
    valid_mask = adj_data.notna() & dft_data.notna()

    if valid_mask.sum() == 0:
        print(f"No valid data pairs for column {col}")
        return None

    diff_data = adj_data - dft_data
    diff_data = diff_data[valid_mask]

    # Create depth label
    depth_label = create_depth_label(col)

    # Create the plot
    p = figure(
        title=f"{depth_label} (ADJ - DFT)",
        x_axis_type="datetime",
        width=500,
        height=350,
        x_axis_label="Date",
        y_axis_label=diff_label,
        toolbar_location="above",
        x_range=shared_x_range
    )

    # Plot the difference data
    p.line(
        diff_data.index,
        diff_data.values,
        line_width=2,
        color='blue',
        alpha=0.8
    )

    # Add a horizontal line at y=0
    p.line(
        [diff_data.index.min(), diff_data.index.max()],
        [0, 0],
        line_width=1,
        color='black',
        alpha=0.5,
        line_dash='dashed'
    )

    # Set minimum y-axis range if differences are very small or zero
    y_min, y_max = diff_data.min(), diff_data.max()
    y_range = y_max - y_min

    # If the range is very small (close to zero differences), set a fixed range
    if y_range < 0.1:  # Adjust threshold as needed
        p.y_range.start = -1.0
        p.y_range.end = 1.0
    else:
        # Add some padding to the natural range
        padding = y_range * 0.1
        p.y_range.start = y_min - padding
        p.y_range.end = y_max + padding

    # Add hover tool
    hover_units = get_hover_units(param_type)
    hover_tooltip = [
        ("Date", "@x{%F %T}"),
        ("Difference", f"@y{{0.000}} {hover_units}")
    ]

    hover = HoverTool(
        tooltips=hover_tooltip,
        formatters={"@x": "datetime"},
        mode='vline'
    )
    p.add_tools(hover)

    # Style the plot
    p.background_fill_color = "#f0f0f0"  # Light grey background
    p.grid.grid_line_alpha = 0.3

    # Format x-axis
    p.xaxis.formatter = DatetimeTickFormatter(
        hours="%Y-%m-%d %H:%M",
        days="%Y-%m-%d",
        months="%Y-%m-%d",
        years="%Y-%m-%d"
    )

    return p


def create_difference_figure(df_adj, df_dft, param_type, diff_label, basename):
    """Create figure with difference subplots for each instrument"""

    # Get common columns
    common_columns = list(set(df_adj.columns) & set(df_dft.columns))

    # Sort columns with SSS first, then by numeric depth
    def sort_columns(col):
        if col == 'SSS':
            return (0, 0)  # SSS comes first
        elif col.startswith('S'):
            try:
                depth = int(col[1:])
                return (1, depth)  # Then by depth
            except:
                return (2, col)  # Unknown format last
        else:
            return (2, col)

    common_columns.sort(key=sort_columns)
    print(f"Creating difference figure with {len(common_columns)} subplots")
    print(f"Column order: {common_columns}")

    if not common_columns:
        print("No common columns between .adj and .dft files")
        return None

    # Create shared x-range for synchronized zooming
    shared_x_range = DataRange1d()

    # Create subplots
    plots = []
    for col in common_columns:
        subplot = create_difference_subplot(df_adj, df_dft, col, param_type, diff_label, shared_x_range)
        if subplot is not None:
            plots.append(subplot)

    if not plots:
        print("No valid subplots created for differences")
        return None

    # Arrange plots in a grid (2 columns)
    grid_plots = []
    for i in range(0, len(plots), 2):
        row = plots[i:i+2]
        # Pad with None if odd number of plots
        if len(row) == 1:
            row.append(None)
        grid_plots.append(row)

    # Create grid layout
    grid = gridplot(grid_plots, sizing_mode='scale_width')

    return grid


def main():
    if len(sys.argv) != 2:
        print("Usage: python atlas_compare_adjustments.py <base_filename_without_extension>")
        print("Example: python atlas_compare_adjustments.py mooring_sal")
        print("This will look for mooring_sal.adj and mooring_sal.dft files")
        print("Creates difference plots (ADJ - DFT) for each instrument/depth")
        print("Supported parameter types: salinity, density, conductivity")
        sys.exit(1)

    base_filename = sys.argv[1]

    # Construct filenames
    adj_filename = f"{base_filename}.adj"
    dft_filename = f"{base_filename}.dft"

    # Check if files exist
    if not os.path.exists(adj_filename):
        print(f"Error: File '{adj_filename}' not found")
        sys.exit(1)

    if not os.path.exists(dft_filename):
        print(f"Error: File '{dft_filename}' not found")
        sys.exit(1)

    # Detect parameter type from base filename
    param_type, diff_label = detect_parameter_type(base_filename)

    print(f"Detected parameter type: {param_type}")
    print(f"Difference label: {diff_label}")
    print(f"Processing files: {adj_filename}, {dft_filename}")

    # Parse both files
    df_adj = parse_dft_file(adj_filename)
    df_dft = parse_dft_file(dft_filename)

    if df_adj is None:
        print("Failed to parse .adj file")
        sys.exit(1)

    if df_dft is None:
        print("Failed to parse .dft file")
        sys.exit(1)

    basename = os.path.basename(base_filename)

    # Create difference figure
    print("\nCreating difference figure...")
    diff_figure = create_difference_figure(df_adj, df_dft, param_type, diff_label, basename)

    if diff_figure is not None:
        # Save as PNG
        png_output_name = f"{basename}_adj-dft.png"
        try:
            export_png(diff_figure, filename=png_output_name, width=1200, height=1000)
            print(f"Difference figure saved to '{png_output_name}'")
        except Exception as e:
            print(f"Warning: Could not save PNG ({e})")

        # Show the plot on screen
        output_file(f"{basename}_adj-dft.html")
        show(diff_figure)
        print(f"Difference figure displayed and saved to '{basename}_adj-dft.html'")

    print(f"\nCompleted analysis for {param_type} data")
    print("Features:")
    print("- Difference subplots: Shows ADJ - DFT differences for each instrument")
    print("- Synchronized zooming across all subplots")
    print("- Hover tooltips show exact values and timestamps")


if __name__ == "__main__":
    main()
