#!/usr/bin/env python3
"""
Atlas comparison plotting script for multiple deployment data files
Plots selected sensors/depths from multiple .adj, .dft, or .flg files
with daily averages subplot below for comparison analysis
Usage: python atlas_plot_compare.py file1 [file2 ...] [--depth DEPTH1,DEPTH2,...] [--files file1:depth1,depth2 file2:depth3,depth4]
"""

import pandas as pd
import sys
import os
import argparse
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, DatetimeTickFormatter, DataRange1d
from bokeh.layouts import column

def detect_parameter_type(filename):
    """Detect parameter type from filename"""
    basename = os.path.basename(filename).lower()

    # Determine parameter type
    if 'sal' in basename:
        param_type = 'salinity'
        y_label = 'Salinity (PSU)'
        invert_y = False
    elif 'temp' in basename:
        param_type = 'temperature'
        y_label = 'Temperature (C)'
        invert_y = False
    elif 'cond' in basename:
        param_type = 'conductivity'
        y_label = 'Conductivity (mS/cm)'
        invert_y = False
    elif 'dens' in basename:
        param_type = 'density'
        y_label = 'Density (ρ₀)'
        invert_y = True
    else:
        param_type = 'unknown'
        y_label = 'Value'
        invert_y = False

    return param_type, y_label, invert_y

def parse_atlas_file(file_path):
    """Parse .dft, .adj, or .flg deployment files (identical format)"""
    file_ext = os.path.splitext(file_path)[1].upper()
    print(f"Reading {file_ext} file: {file_path}")

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
                        # Filter out bad values - adjust ranges for different parameters
                        if val > 1000 or val < -100:  # More generous range for density/conductivity
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

        return df

    except Exception as e:
        print(f"Error parsing {file_ext} file: {e}")
        return None

def create_depth_label(col, full_filename=None):
    """Create appropriate label for depth column with optional full filename identifier"""
    if col == 'SSS':
        base_label = '1m (SSS)'
    elif col.startswith('S'):
        try:
            depth = int(col[1:])
            base_label = f'{depth}m'
        except:
            base_label = col
    else:
        base_label = col

    if full_filename:
        return f"{base_label} ({full_filename})"
    else:
        return base_label

def get_hover_units(param_type):
    """Get appropriate units for hover tooltips"""
    if param_type == 'salinity':
        return 'PSU'
    elif param_type == 'temperature':
        return 'C'
    elif param_type == 'conductivity':
        return 'mS/cm'
    elif param_type == 'density':
        return 'ρ₀'
    else:
        return ''

def calculate_daily_averages(df):
    """Calculate daily averages for all columns"""
    # Resample to daily averages
    daily_df = df.resample('D').mean()
    return daily_df

def filter_dataframe_by_depths(df, requested_depths, file_basename):
    """Filter dataframe to only include specified depths"""
    if requested_depths is None:
        return df

    # Convert requested depths to column names
    requested_columns = []
    for depth in requested_depths:
        if depth == 1:
            requested_columns.append('SSS')
        else:
            requested_columns.append(f'S{depth:03d}')

    # Filter to only include requested columns that exist in the dataframe
    available_columns = [col for col in requested_columns if col in df.columns]

    if not available_columns:
        print(f"Warning: None of the requested depths {requested_depths} found in {file_basename}")
        available_depths = []
        for col in df.columns:
            if col == 'SSS':
                available_depths.append('1')
            elif col.startswith('S'):
                try:
                    depth = int(col[1:])
                    available_depths.append(str(depth))
                except:
                    available_depths.append(col)
        print(f"Available depths in {file_basename}: {available_depths}")
        return pd.DataFrame()

    print(f"Filtering {file_basename} to depths: {requested_depths}")
    return df[available_columns]



def parse_file_depth_specification(files_spec):
    """Parse file:depth1,depth2 specifications"""
    file_depth_map = {}

    for spec in files_spec:
        if ':' in spec:
            file_part, depth_part = spec.split(':', 1)
            depths = []
            for depth in depth_part.split(','):
                try:
                    depths.append(int(depth.strip()))
                except ValueError:
                    print(f"Warning: Invalid depth '{depth.strip()}' in specification '{spec}'")
            if depths:
                file_depth_map[file_part] = depths
        else:
            # No depths specified for this file, will use global depth filter
            file_depth_map[spec] = None

    return file_depth_map

def create_comparison_plot_figure(all_data, param_type, y_label, shared_x_range=None, invert_y=False, is_daily=False):
    """Create comparison plot with data from multiple files"""

    plot_type = "Daily Averages" if is_daily else "Raw Data"
    title = f"{param_type.title()} {plot_type} - Multi-File Comparison"

    # Create the plot
    p = figure(
        title=title,
        x_axis_type="datetime",
        width=1200,
        height=500,
        x_axis_label="Date",
        y_axis_label=y_label + (" (Daily Average)" if is_daily else ""),
        toolbar_location="above",
        x_range=shared_x_range,
        tools="pan,wheel_zoom,box_zoom,reset,save,fullscreen",
        sizing_mode="stretch_both",
        max_width=2000,
        max_height=800
    )

    # Extended color palette for multiple files/depths
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]

    # Line styles for variety when colors repeat
    line_styles = ['solid', 'dashed', 'dotted', 'dotdash', 'dashdot']

    color_idx = 0
    line_style_idx = 0

    # Plot data from each file
    for file_info in all_data:
        file_basename = file_info['basename']
        df = file_info['data']

        # Get all columns and sort them
        columns = list(df.columns)

        # Sort columns with SSS first, then by numeric depth
        def sort_columns(col):
            if col == 'SSS':
                return (0, 0)
            elif col.startswith('S'):
                try:
                    depth = int(col[1:])
                    return (1, depth)
                except:
                    return (2, col)
            else:
                return (2, col)

        columns.sort(key=sort_columns)

        # Plot each depth for this file
        for col in columns:
            if col in df.columns:
                clean_data = df[col].dropna()
                if len(clean_data) > 0:
                    depth_label = create_depth_label(col, file_info['full_filename'])

                    line_width = 3 if is_daily else 2
                    alpha = 0.9 if is_daily else 0.8

                    p.line(
                        clean_data.index,
                        clean_data.values,
                        line_width=line_width,
                        color=colors[color_idx % len(colors)],
                        line_dash=line_styles[line_style_idx % len(line_styles)],
                        alpha=alpha,
                        legend_label=depth_label
                    )

                    color_idx += 1
                    if color_idx % len(colors) == 0:
                        line_style_idx += 1

    # Customize legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.9
    p.legend.border_line_color = "black"
    p.legend.border_line_width = 1

    # Add hover tool
    hover_units = get_hover_units(param_type)
    if is_daily:
        hover_tooltip = [
            ("Date", "@x{%F}"),
            ("Daily Average", f"@y{{0.000}} {hover_units}")
        ]
    else:
        hover_tooltip = [
            ("Date", "@x{%F %T}"),
            ("Value", f"@y{{0.000}} {hover_units}")
        ]

    hover = HoverTool(
        tooltips=hover_tooltip,
        formatters={"@x": "datetime"},
        mode='vline'
    )
    p.add_tools(hover)

    # Make hover tool inactive by default
    p.toolbar.active_inspect = []

    # Style the plot
    p.background_fill_color = "#e6e6e6"
    p.grid.grid_line_alpha = 0.3

    # Format x-axis
    if is_daily:
        p.xaxis.formatter = DatetimeTickFormatter(
            days="%Y-%m-%d",
            months="%Y-%m-%d",
            years="%Y-%m-%d"
        )
    else:
        p.xaxis.formatter = DatetimeTickFormatter(
            hours="%Y-%m-%d %H:%M",
            days="%Y-%m-%d",
            months="%Y-%m-%d",
            years="%Y-%m-%d"
        )

    # Invert y-axis for density plots
    if invert_y:
        p.y_range.flipped = True

    return p

def parse_depths(depth_str):
    """Parse depth string into list of integers"""
    if not depth_str:
        return None

    depths = []
    for depth in depth_str.split(','):
        try:
            depths.append(int(depth.strip()))
        except ValueError:
            print(f"Warning: Invalid depth value '{depth.strip()}' ignored")

    return depths if depths else None

def main():
    parser = argparse.ArgumentParser(
        description="Atlas comparison plotting script for multiple deployment data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all depths from two files
  python atlas_plot_compare.py file1.adj file2.adj

  # Plot specific depths from multiple files (same depths for all files)
  python atlas_plot_compare.py file1.adj file2.adj --depth 10,40

  # Plot different depths from different files
  python atlas_plot_compare.py --file-depths file1.adj:10,40 file2.adj:1,20

  # Mix approaches - specific files with depths, plus additional files with global depth filter
  python atlas_plot_compare.py file3.adj --file-depths file1.adj:10,40 file2.adj:1,20 --depth 30

Supported file types: .adj, .dft, .flg
Supported parameter types: salinity, temperature, density, conductivity

Note: All files should contain the same parameter type for meaningful comparison.
        """
    )

    parser.add_argument('files', nargs='*', help='Input file paths (.adj, .dft, or .flg)')
    parser.add_argument('--depth', type=str, help='Comma-separated list of depths to plot from all files (e.g., "40" or "10,40")')
    parser.add_argument('--file-depths', nargs='*', help='File-specific depth specifications (e.g., "file1.adj:10,40" "file2.adj:1,20")')

    args = parser.parse_args()

    # Validate arguments
    if not args.files and not args.file_depths:
        parser.print_help()
        sys.exit(1)

    # Parse file and depth specifications
    file_depth_map = {}
    global_depths = parse_depths(args.depth)

    # Handle --file-depths argument (file:depth specifications)
    if args.file_depths:
        file_depth_map.update(parse_file_depth_specification(args.file_depths))

    # Handle positional file arguments (use global depth filter)
    for filename in args.files:
        if filename not in file_depth_map:
            file_depth_map[filename] = global_depths

    if not file_depth_map:
        print("Error: No files specified")
        parser.print_help()
        sys.exit(1)

    # Validate all files exist
    for filename in file_depth_map.keys():
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found")
            sys.exit(1)

    # Check file extensions
    param_types = set()
    for filename in file_depth_map.keys():
        file_ext = os.path.splitext(filename)[1]
        if file_ext.lower() not in ['.adj', '.dft', '.flg']:
            print(f"Error: Unsupported file type '{file_ext}' for file '{filename}'")
            print("Supported file types: .adj, .dft, .flg")
            sys.exit(1)

        # Check parameter type consistency
        param_type, _, _ = detect_parameter_type(filename)
        param_types.add(param_type)

    if len(param_types) > 1:
        print(f"Warning: Multiple parameter types detected: {param_types}")
        print("For best comparison, all files should contain the same parameter type")

    # Use the first file's parameter type for plot settings
    first_file = next(iter(file_depth_map.keys()))
    param_type, y_label, invert_y = detect_parameter_type(first_file)

    # Parse all files and apply depth filtering
    all_data = []

    for filename, requested_depths in file_depth_map.items():
        print(f"\nProcessing {filename}...")

        # Parse the file
        df = parse_atlas_file(filename)

        if df is None:
            print(f"Failed to parse file: {filename}")
            continue

        basename = os.path.splitext(os.path.basename(filename))[0]

        # Apply depth filtering
        if requested_depths is not None:
            # Filter to specific depths for this file
            filtered_df = filter_dataframe_by_depths(df, requested_depths, basename)
            if filtered_df.empty:
                print(f"No valid data after filtering {filename}")
                continue
            df = filtered_df

        # Calculate daily averages
        df_daily = calculate_daily_averages(df)

        all_data.append({
            'filename': filename,
            'basename': basename,
            'full_filename': os.path.basename(filename),
            'data': df,
            'daily': df_daily,
            'depths': requested_depths
        })

    if not all_data:
        print("No valid data found in any files")
        sys.exit(1)

    # Create shared x-range for synchronized zooming
    shared_x_range = DataRange1d()

    # Create main comparison figure
    main_figure = create_comparison_plot_figure(all_data, param_type, y_label, shared_x_range, invert_y, is_daily=False)

    if main_figure is None:
        print("Failed to create main figure")
        sys.exit(1)

    # Create daily average comparison figure
    daily_data = []
    for item in all_data:
        daily_data.append({
            'filename': item['filename'],
            'basename': item['basename'],
            'full_filename': item['full_filename'],
            'data': item['daily'],
            'depths': item['depths']
        })

    daily_figure = create_comparison_plot_figure(daily_data, param_type, y_label, shared_x_range, invert_y, is_daily=True)

    if daily_figure is None:
        print("Failed to create daily average figure")
        sys.exit(1)

    # Combine plots in a column layout
    combined_figure = column(main_figure, daily_figure, sizing_mode="stretch_both")

    # Generate output filename
    file_basenames = [item['basename'] for item in all_data]
    if len(file_basenames) == 1:
        basename_part = file_basenames[0]
    elif len(file_basenames) <= 3:
        basename_part = '_vs_'.join(file_basenames)
    else:
        basename_part = f"{file_basenames[0]}_vs_{len(file_basenames)-1}others"

    # Add depth information if global depths were used
    depth_suffix = ""
    if global_depths:
        depth_suffix = f"_depths_{'_'.join(map(str, global_depths))}"

    html_output_name = f"{param_type}_comparison_{basename_part}{depth_suffix}.html"
    output_file(html_output_name)
    show(combined_figure)
    print(f"\nComparison figure saved to '{html_output_name}'")



if __name__ == "__main__":
    main()
