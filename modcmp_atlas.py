#!/usr/bin/env python3
"""
File comparison plotting script similar to modcmp_bokeh.py but using file inputs
Creates side-by-side comparison plots of two data files (like .adj and .dft files)
Usage: python filecmp_bokeh.py <file1> <file2>
Example: python filecmp_bokeh.py mooring_sal.adj mooring_sal.dft
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse
#from pathlib import Path
from bokeh.plotting import figure, show
from bokeh.layouts import column as bokeh_column, row
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, DatetimeTickFormatter, Div
from bokeh.palettes import Category10


def detect_parameter_type(filename):
    """Detect parameter type from filename"""
    basename = os.path.basename(filename).lower()

    # Determine parameter type
    if 'sal' in basename:
        param_type = 'salinity'
        y_label = 'Salinity (PSU)'
        invert_y = False
    elif 'cond' in basename:
        param_type = 'conductivity'
        y_label = 'Conductivity (mS/cm)'
        invert_y = False
    elif 'dens' in basename:
        param_type = 'density'
        y_label = 'Potential Density (ρ₀)'
        invert_y = True  # Density plots are often inverted
    else:
        param_type = 'unknown'
        y_label = 'Value'
        invert_y = False

    return param_type, y_label, invert_y

def parse_data_file(file_path):
    """Parse deployment data files (.adj, .dft, or similar formats)"""
    file_ext = os.path.splitext(file_path)[1]
    print(f"Reading {file_ext.upper()} file: {file_path}")

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Skip first 3 lines, get headers from lines 4-5
        if len(lines) < 6:
            print(f"File {file_path} has insufficient lines")
            return None

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
                            val = np.nan
                        values.append(val)
                    else:
                        values.append(np.nan)

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
        print(f"Columns: {list(df.columns)}")

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

def main():
    parser = argparse.ArgumentParser(description='Compare two deployment data files with side-by-side plots')
    parser.add_argument("file1", help="First data file (e.g., pt006_sal.adj)")
    parser.add_argument("file2", help="Second data file (e.g., pt006_sal.dft)")
    args = parser.parse_args()

    file1_path = args.file1
    file2_path = args.file2

    # Check if files exist
    if not os.path.exists(file1_path):
        print(f"Error: File '{file1_path}' not found")
        sys.exit(1)

    if not os.path.exists(file2_path):
        print(f"Error: File '{file2_path}' not found")
        sys.exit(1)

    # Detect parameter type from first filename
    param_type, y_label, invert_y = detect_parameter_type(file1_path)
    print(f"Detected parameter type: {param_type}")

    # Parse both files
    df1 = parse_data_file(file1_path)
    df2 = parse_data_file(file2_path)

    if df1 is None or df2 is None:
        print("Failed to parse one or both files")
        sys.exit(1)

    # Get file names for titles
    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)

    # Ensure both dataframes have datetime index
    df1.index = pd.to_datetime(df1.index).tz_localize(None)
    df2.index = pd.to_datetime(df2.index).tz_localize(None)

    # Get common columns
    common_columns = list(set(df1.columns) & set(df2.columns))

    if not common_columns:
        print("No common columns found between files")
        sys.exit(1)

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

    common_columns.sort(key=sort_columns)
    print(f"Common columns: {common_columns}")

    # Prepare plotting dataframes
    df1_plot = df1[common_columns].copy()
    df2_plot = df2[common_columns].copy()

    # Calculate y-range for both datasets
    y_min = min(df1_plot.min().min(), df2_plot.min().min())
    y_max = max(df1_plot.max().max(), df2_plot.max().max())
    y_range = (y_min - 0.1, y_max + 0.1)  # Add a small buffer

    # Calculate x-range for both datasets
    x_range1 = (df1_plot.index.min(), df1_plot.index.max())
    x_range2 = (df2_plot.index.min(), df2_plot.index.max())

    # Create Bokeh figures with increased width and height
    p1 = figure(title=file1_name, x_axis_label='Date', y_axis_label=y_label,
               x_range=x_range1, y_range=y_range,
               width=800, height=600, x_axis_type="datetime",
               toolbar_location="left", background_fill_color="lightgrey")
    p2 = figure(title=file2_name, x_axis_label='Date', y_axis_label=y_label,
               x_range=x_range2, y_range=y_range,
               width=800, height=600, x_axis_type="datetime",
               background_fill_color="lightgrey")

    # Link the y-ranges of both plots
    p1.y_range = p2.y_range

    # Set x-axis formatter to ISO format
    date_formatter = DatetimeTickFormatter(
        hours="%Y-%m-%d %H:%M",
        days="%Y-%m-%d",
        months="%Y-%m-%d",
        years="%Y-%m-%d"
    )
    p1.xaxis.formatter = date_formatter
    p2.xaxis.formatter = date_formatter

    # Generate color palettes using Category10
    num_colors = len(common_columns)
    colors = Category10[10] * (num_colors // 10 + 1)  # Repeat the palette if needed
    colors = colors[:num_colors]  # Trim to the required number of colors

    # Store all glyph renderers
    renderers1 = []
    renderers2 = []

    # Add lines to the first plot - always create a renderer for each column
    for column, color in zip(common_columns, colors):
        clean_data1 = df1_plot[column].dropna()
        depth_label = create_depth_label(column)

        if len(clean_data1) > 0:
            # Create renderer with actual data
            source = ColumnDataSource(data=dict(
                x=clean_data1.index,
                y=clean_data1.values
            ))
            renderer = p1.line('x', 'y', source=source, color=color,
                             name=str(column), legend_label=depth_label)
        else:
            # Create empty renderer that still shows in legend
            source = ColumnDataSource(data=dict(x=[], y=[]))
            renderer = p1.line('x', 'y', source=source, color=color,
                             name=str(column), legend_label=depth_label,
                             visible=False)  # Make invisible but keep in legend

        renderers1.append(renderer)

    # Add lines to the second plot - always create a renderer for each column
    for column, color in zip(common_columns, colors):
        clean_data2 = df2_plot[column].dropna()
        depth_label = create_depth_label(column)

        if len(clean_data2) > 0:
            # Create renderer with actual data
            source = ColumnDataSource(data=dict(
                x=clean_data2.index,
                y=clean_data2.values
            ))
            renderer = p2.line('x', 'y', source=source, color=color,
                             legend_label=depth_label, name=str(column))
        else:
            # Create empty renderer that still shows in legend
            source = ColumnDataSource(data=dict(x=[], y=[]))
            renderer = p2.line('x', 'y', source=source, color=color,
                             legend_label=depth_label, name=str(column),
                             visible=False)  # Make invisible but keep in legend

        renderers2.append(renderer)

    # Add hover tool
    hover = HoverTool(tooltips=[('Date', '@x{%F %T}'), (y_label, '@y{0.0000}')],
                     formatters={'@x': 'datetime'})
    p1.add_tools(hover)
    p2.add_tools(hover)

    # Customize the plots
    p1.legend.location = "left"
    p1.legend.click_policy = "hide"
    p2.legend.location = "right"
    p2.legend.click_policy = "hide"

    # Create a custom JavaScript callback for syncing zooming and panning
    sync_callback = CustomJS(args=dict(p1=p1, p2=p2), code="""
        const y_range = p2.y_range;
        p1.y_range.start = y_range.start;
        p1.y_range.end = y_range.end;
    """)

    # Attach the callback to p2's y-range
    p2.y_range.js_on_change('start', sync_callback)
    p2.y_range.js_on_change('end', sync_callback)

    # Create a custom JavaScript callback for toggling visibility
    visibility_code = """
        const renderers1 = Bokeh.documents[0].get_model_by_name('plot1').renderers;
        const renderers2 = Bokeh.documents[0].get_model_by_name('plot2').renderers;

        for (let i = 0; i < renderers1.length; i++) {
            if (renderers1[i].name === this.name) {
                renderers1[i].visible = this.visible;
            }
        }

        for (let i = 0; i < renderers2.length; i++) {
            if (renderers2[i].name === this.name) {
                renderers2[i].visible = this.visible;
            }
        }
    """

    # Attach the visibility callback to each renderer in the second plot
    for renderer in renderers2:
        renderer.js_on_change('visible', CustomJS(code=visibility_code))

    # Set names for the plots
    p1.name = 'plot1'
    p2.name = 'plot2'

    # Calculate differences between last non-NaN value in df1 and first non-NaN value in df2
    differences = {}
    for column in common_columns:
        last_value_df1 = df1_plot[column].dropna().iloc[-1] if not df1_plot[column].dropna().empty else None
        first_value_df2 = df2_plot[column].dropna().iloc[0] if not df2_plot[column].dropna().empty else None

        if last_value_df1 is not None and first_value_df2 is not None:
            difference = last_value_df1 - first_value_df2
            differences[column] = difference
        else:
            differences[column] = None  # No data available for difference calculation

    # Create HTML content for differences table
    html_content = f"""
    <div style="width: 100%; padding: 20px; font-family: Arial, sans-serif;">
        <h3 style="text-align: center; margin-bottom: 20px; color: #333;">
            Differences between {file1_name} and {file2_name}
        </h3>
        <p style="text-align: center; margin-bottom: 15px; color: #666; font-style: italic;">
            Last non-NaN in {file1_name} - First non-NaN in {file2_name}
        </p>
        <table style="margin: 0 auto; border-collapse: collapse; width: 60%; max-width: 500px;">
            <thead>
                <tr style="background-color: #f0f0f0;">
                    <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Depth</th>
                    <th style="border: 1px solid #ddd; padding: 12px; text-align: right;">Difference</th>
                </tr>
            </thead>
            <tbody>
    """

    for column in common_columns:
        depth_label = create_depth_label(column)
        diff = differences.get(column)

        if diff is not None:
            sign = "+" if diff >= 0 else ""
            color = "green" if diff >= 0 else "red"
            diff_display = f"{sign}{diff:.6f}"
        else:
            color = "#666"  # Gray color for no data
            diff_display = "-"

        html_content += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 10px;">{depth_label}</td>
                    <td style="border: 1px solid #ddd; padding: 10px; text-align: right; color: {color}; font-weight: bold;">
                        {diff_display}
                    </td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </div>
    """

    # Create Div widget to display the differences
    differences_div = Div(text=html_content, width=1600, height=200)

    # Create layout with plots on top and differences below
    plots_row = row(p1, p2)
    layout = bokeh_column(plots_row, differences_div)

    # Show the complete layout
    show(layout)
    print(f"\nCompleted comparison of {file1_name} and {file2_name}")
    print("Features:")
    print("- Side-by-side comparison plots")
    print("- Synchronized y-axis zooming")
    print("- Interactive legend (click to hide/show lines)")
    print("- Hover tooltips with exact values")
    print("- Difference calculations displayed below plots")

    # Print differences table to screen
    print(f"\nDifferences between {file1_name} and {file2_name}")
    print(f"Last non-NaN in {file1_name} - First non-NaN in {file2_name}:")
    print("-" * 50)
    for column in common_columns:
        depth_label = create_depth_label(column)
        diff = differences.get(column)
        if diff is not None:
            print(f"{depth_label:>10}: {diff:11.6f}")
        else:
            print(f"{depth_label:>10}: {'-':>11}")
    print("-" * 50)

if __name__ == "__main__":
    main()
