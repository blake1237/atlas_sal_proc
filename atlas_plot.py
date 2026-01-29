#!/usr/bin/env python3
"""
Atlas plotting script for single deployment data file
Plots all sensors/depths from a single .adj, .dft, or .flg file
with daily averages subplot below
Usage: python atlas_plot.py <filename> [--depth DEPTH1,DEPTH2,...]
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

def filter_dataframe_by_depths(df, requested_depths):
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
        print(f"Warning: None of the requested depths {requested_depths} found in data")
        print(f"Available depths: {[col for col in df.columns]}")
        return df

    print(f"Filtering to depths: {requested_depths}")
    return df[available_columns]

def create_single_plot_figure(df, param_type, y_label, basename, file_type, shared_x_range=None, invert_y=False):
    """Create single plot with all instruments/depths"""

    # Get all columns
    columns = list(df.columns)

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

    columns.sort(key=sort_columns)

    # Create the main plot
    p = figure(
        title=f"{param_type.title()} Data - {file_type.upper()} File ({basename})",
        x_axis_type="datetime",
        width=1200,
        height=500,
        x_axis_label="Date",
        y_axis_label=y_label,
        toolbar_location="above",
        x_range=shared_x_range,
        tools="pan,wheel_zoom,box_zoom,reset,save,fullscreen",
        sizing_mode="stretch_both",
        max_width=2000,
        max_height=800
    )

    # Use custom color palette excluding brown
    # Category10 colors: Blue, Orange, Green, Red, Purple, Brown, Pink, Gray, Olive, Cyan
    # Excluding brown (index 5)
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Repeat colors if we have more sensors than colors
    colors = custom_colors * (len(columns) // len(custom_colors) + 1)

    # Plot each instrument/depth
    valid_columns = []
    for i, col in enumerate(columns):
        if col in df.columns:
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                depth_label = create_depth_label(col)

                p.line(
                    clean_data.index,
                    clean_data.values,
                    line_width=2,
                    color=colors[i % len(colors)],
                    alpha=0.8,
                    legend_label=depth_label
                )
                valid_columns.append(col)
            else:
                print(f"No valid data for column {col}")

    if not valid_columns:
        print("No valid data found in any columns")
        return None

    # Customize legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.9
    p.legend.border_line_color = "black"
    p.legend.border_line_width = 1

    # Add hover tool
    hover_units = get_hover_units(param_type)
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

def create_daily_average_plot(df_daily, param_type, y_label, basename, file_type, shared_x_range, invert_y=False):
    """Create daily average plot with all instruments/depths"""

    # Get all columns
    columns = list(df_daily.columns)

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

    columns.sort(key=sort_columns)

    # Create the daily average plot
    p = figure(
        title=f"Daily Averages - {param_type.title()} Data",
        x_axis_type="datetime",
        width=1200,
        height=500,
        x_axis_label="Date",
        y_axis_label=f"{y_label} (Daily Average)",
        toolbar_location="above",
        x_range=shared_x_range,
        tools="pan,wheel_zoom,box_zoom,reset,save,fullscreen",
        sizing_mode="stretch_both",
        max_width=2000,
        max_height=800
    )

    # Use same custom color palette
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Repeat colors if we have more sensors than colors
    colors = custom_colors * (len(columns) // len(custom_colors) + 1)

    # Plot each instrument/depth daily average
    valid_columns = []
    for i, col in enumerate(columns):
        if col in df_daily.columns:
            clean_data = df_daily[col].dropna()
            if len(clean_data) > 0:
                depth_label = create_depth_label(col)

                p.line(
                    clean_data.index,
                    clean_data.values,
                    line_width=3,  # Slightly thicker lines for daily averages
                    color=colors[i % len(colors)],
                    alpha=0.9,
                    legend_label=depth_label
                )

                valid_columns.append(col)

    if not valid_columns:
        print("No valid daily average data found")
        return None

    # Customize legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.9
    p.legend.border_line_color = "black"
    p.legend.border_line_width = 1

    # Add hover tool
    hover_units = get_hover_units(param_type)
    hover_tooltip = [
        ("Date", "@x{%F}"),
        ("Daily Average", f"@y{{0.000}} {hover_units}")
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
    p.background_fill_color = "#e6e6e6"  # Light grey background
    p.grid.grid_line_alpha = 0.3

    # Format x-axis
    p.xaxis.formatter = DatetimeTickFormatter(
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
        description="Atlas plotting script for single deployment data file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python atlas_plot.py mooring_sal.adj                    # Plot all depths
  python atlas_plot.py mooring_sal.adj --depth 40         # Plot only 40m depth
  python atlas_plot.py mooring_sal.adj --depth 10,40      # Plot 10m and 40m depths
  python atlas_plot.py mooring_sal.adj --depth 1,10,40    # Plot 1m (SSS), 10m, and 40m depths

Supported file types: .adj, .dft, .flg
Supported parameter types: salinity, temperature, density, conductivity
        """
    )

    parser.add_argument('filename', help='Input file path (.adj, .dft, or .flg)')
    parser.add_argument('--depth', type=str, help='Comma-separated list of depths to plot (e.g., "40" or "10,40")')

    args = parser.parse_args()
    filename = args.filename
    requested_depths = parse_depths(args.depth)

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        sys.exit(1)

    # Get file extension and basename
    file_ext = os.path.splitext(filename)[1]
    basename = os.path.splitext(os.path.basename(filename))[0]

    # Check if supported file type
    if file_ext.lower() not in ['.adj', '.dft', '.flg']:
        print(f"Error: Unsupported file type '{file_ext}'")
        print("Supported file types: .adj, .dft, .flg")
        sys.exit(1)

    # Detect parameter type from filename
    param_type, y_label, invert_y = detect_parameter_type(filename)

    # Parse the file
    df = parse_atlas_file(filename)

    if df is None:
        print("Failed to parse file")
        sys.exit(1)

    # Filter by requested depths if specified
    df = filter_dataframe_by_depths(df, requested_depths)

    # Calculate daily averages
    df_daily = calculate_daily_averages(df)

    # Create shared x-range for synchronized zooming
    shared_x_range = DataRange1d()

    # Create main figure
    main_figure = create_single_plot_figure(df, param_type, y_label, basename, file_ext, shared_x_range, invert_y)

    if main_figure is None:
        print("Failed to create main figure")
        sys.exit(1)

    # Create daily average figure
    daily_figure = create_daily_average_plot(df_daily, param_type, y_label, basename, file_ext, shared_x_range, invert_y)

    if daily_figure is None:
        print("Failed to create daily average figure")
        sys.exit(1)

    # Combine plots in a column layout with responsive sizing
    combined_figure = column(main_figure, daily_figure, sizing_mode="stretch_both")

    # Save as HTML only
    depth_suffix = ""
    if requested_depths:
        depth_suffix = f"_depths_{'_'.join(map(str, requested_depths))}"
    html_output_name = f"{param_type}_{file_ext[1:]}_{basename}{depth_suffix}.html"
    output_file(html_output_name)
    show(combined_figure)
    print(f"Figure saved to '{html_output_name}'")

if __name__ == "__main__":
    main()
