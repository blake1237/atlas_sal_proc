import argparse
import pandas as pd
import os
from bokeh.plotting import figure
from bokeh.layouts import column as bokeh_column, row
from bokeh.models import ColumnDataSource, HoverTool, Button, Span, Div
from datetime import datetime
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import sys

def validate_column_name(column_name):
    # This will be populated dynamically after parsing the file
    return column_name

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot deployment file data with interactive difference plots.')
parser.add_argument('dep_file', help='Path to deployment file')
parser.add_argument('inst_col', help='Instrument column name (will be validated after file parsing)')
args = parser.parse_args()

dep_file = args.dep_file
inst_col = args.inst_col
dep_name = os.path.splitext(os.path.basename(dep_file))[0]

# Extract just the mooring segment (e.g., "268a" from "sal268a")
if dep_name.startswith('sal'):
    mooring_segment = dep_name[3:]  # Remove "sal" prefix
elif dep_name.startswith('cond'):
    mooring_segment = dep_name[4:]  # Remove "cond" prefix
elif dep_name.startswith('temp'):
    mooring_segment = dep_name[4:]  # Remove "temp" prefix
else:
    mooring_segment = dep_name  # Use as-is if no recognized prefix

# Function to extract depth number from column name
def get_depth_number(column):
    if column == 'SSS':
        return -1  # Put SSS first (but it's the smallest depth)
    try:
        return int(column.replace('S', ''))
    except:
        return float('inf')  # Put non-standard names last

# Function to generate column combinations with greater depth first
def generate_column_order(columns):
    base_columns = sorted([col for col in columns if '-' not in col],
                        key=get_depth_number)

    ordered_combinations = []

    # Create all unique differences with greater depth first
    # This avoids duplicates like having both "S010-SSS" and "SSS-S010"
    for i, col1 in enumerate(base_columns):
        for j, col2 in enumerate(base_columns):
            if i != j:
                # Determine which has greater depth
                depth1 = get_depth_number(col1)
                depth2 = get_depth_number(col2)

                # Put greater depth first (but SSS is special case with depth -1)
                if depth1 > depth2:
                    diff_col = f'{col1}-{col2}'
                elif depth2 > depth1:
                    diff_col = f'{col2}-{col1}'
                else:
                    # Same depth (shouldn't happen), use alphabetical
                    continue  # Skip same depths

                # Only add if it exists and we haven't added it yet
                if diff_col in columns and diff_col not in ordered_combinations:
                    ordered_combinations.append(diff_col)

    return ordered_combinations

def create_column_name_from_depth(depth_str):
    """Convert depth string to standard column name"""
    try:
        depth_int = int(depth_str)
        if depth_int == 1:
            return 'SSS'
        else:
            return f'S{depth_int:03d}'
    except:
        # If we can't parse as integer, create a generic name
        return f'COL_{depth_str}'

def parse_deployment_file(file_path):
    """Parse deployment file with flexible column detection"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        print(f"Reading {len(lines)} lines from {file_path}")

        # Skip first 3 lines
        data_lines = lines[3:]

        if len(data_lines) < 2:
            raise ValueError("File doesn't have enough header lines")

        # Parse header lines (instrument serial numbers and depths)
        header_line1 = data_lines[0].strip()  # Instrument serial numbers
        header_line2 = data_lines[1].strip()  # Depths

        print(f"Header line 1: {header_line1}")
        print(f"Header line 2: {header_line2}")

        # Extract instrument serial numbers (skip DATE column)
        header1_parts = header_line1.split()
        # Find the index where DATE appears (could be 'DATE' or similar)
        date_index = -1
        for i, part in enumerate(header1_parts):
            if 'DATE' in part.upper():
                date_index = i
                break

        if date_index == -1:
            # If no DATE found, assume first column is date
            date_index = 0

        # Get instrument serial numbers (skip DATE and quality columns)
        inst_serials = []
        for part in header1_parts[date_index + 1:]:
            if part not in ['QQQQQQ', 'SSSSSS'] and not part.upper().startswith('Q') and not part.upper().startswith('S' * 6):
                inst_serials.append(part)

        # Extract depths from second header line
        header2_parts = header_line2.split()

        # Skip elements until we find numeric values (depths)
        depths = []
        for part in header2_parts:
            try:
                # Try to convert to int - if successful, it's likely a depth
                depth_val = int(part)
                if depth_val > 0 and depth_val < 10000:  # Reasonable depth range
                    depths.append(str(depth_val))
            except ValueError:
                continue

        # If we didn't find any valid depths, try to extract from positions
        if not depths:
            print("No depths found in header2, trying to extract from data positions...")
            # Look at the first data line to determine number of columns
            if len(data_lines) > 2:
                first_data_line = data_lines[2].strip().split()
                num_data_cols = len(first_data_line) - 1  # Subtract 1 for date column
                # Create default depths
                depths = [str(i+1) for i in range(num_data_cols)]

        print(f"Instrument serials: {inst_serials}")
        print(f"Depths detected: {depths}")

        # Create column mapping from depth to standard names
        depth_to_column = {}
        for depth in depths:
            depth_to_column[depth] = create_column_name_from_depth(depth)

        print(f"Column mapping: {depth_to_column}")

        # Parse data lines (starting from line 2, after headers)
        data_rows = []
        skipped_lines = 0

        for line_num, line in enumerate(data_lines[2:], start=6):  # Start from line 6 in original file
            parts = line.strip().split()
            if len(parts) < 2:  # Skip empty lines
                continue

            # Extract date and convert to datetime
            date_str = parts[0]
            try:
                # Try different date formats
                if len(date_str) == 13:  # YYYYDDDHHMMSS
                    dt = pd.to_datetime(date_str, format='%Y%j%H%M%S')
                elif len(date_str) == 12:  # YYYYMMDDHHNN
                    dt = pd.to_datetime(date_str, format='%Y%m%d%H%M')
                elif len(date_str) == 10:  # YYYYMMDDHH
                    dt = pd.to_datetime(date_str, format='%Y%m%d%H')
                else:
                    # Try pandas automatic parsing
                    dt = pd.to_datetime(date_str)
            except Exception as e:
                skipped_lines += 1
                if skipped_lines <= 5:  # Only print first few errors
                    print(f"Warning: Could not parse date '{date_str}' on line {line_num}: {e}")
                continue

            # Create row data
            row_data = {'DATE': dt}

            # Extract salinity values - take as many as we have depths defined
            sal_values = parts[1:]  # All values after date

            # Map values to columns based on position
            for i, depth in enumerate(depths):
                col_name = depth_to_column[depth]
                if i < len(sal_values):
                    val = sal_values[i]
                    try:
                        float_val = float(val)
                        # Handle bad data values (common bad flags)
                        if (val in ['1e35', '1E35', '-999', '999', 'NaN', 'nan'] or
                            float_val > 100 or float_val < 0 or float_val == 999 or float_val == -999):
                            row_data[col_name] = None
                        else:
                            row_data[col_name] = float_val
                    except (ValueError, OverflowError):
                        row_data[col_name] = None
                else:
                    row_data[col_name] = None

            data_rows.append(row_data)

        if skipped_lines > 5:
            print(f"Total skipped lines due to date parsing errors: {skipped_lines}")

        if not data_rows:
            raise ValueError("No valid data rows found")

        # Create DataFrame
        df = pd.DataFrame(data_rows)
        df.set_index('DATE', inplace=True)

        # Remove rows where all values are NaN
        df = df.dropna(how='all')

        # Get final column list
        final_columns = [col for col in df.columns if col != 'DATE']

        print(f"Created DataFrame with shape: {df.shape}")
        print(f"Base columns detected: {final_columns}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        # Show data statistics
        print(f"\nData sample (first 5 rows):")
        print(df.head())

        return df, final_columns

    except Exception as e:
        print(f"Error parsing deployment file: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Load and process data
print(f"Reading deployment file: {dep_file}")
saldf, detected_columns = parse_deployment_file(dep_file)

# Validate the inst_col argument now that we know what columns exist
if inst_col not in detected_columns:
    print(f"Error: Instrument column '{inst_col}' not found in detected columns: {detected_columns}")
    print("Please use one of the detected column names.")
    sys.exit(1)

print(f"Validated instrument column: {inst_col}")

# Use detected columns as base columns
base_columns = [col for col in detected_columns if '-' not in col]
print(f"Base columns: {base_columns}")

# Create UNIQUE difference columns with greater depth first
print("Creating unique difference columns with greater depth first...")
diff_count = 0
for i, col1 in enumerate(base_columns):
    for j, col2 in enumerate(base_columns):
        if i != j:
            # Determine which has greater depth
            depth1 = get_depth_number(col1)
            depth2 = get_depth_number(col2)

            # Create difference with greater depth first
            if depth1 > depth2:
                diff_col = f'{col1}-{col2}'
                if diff_col not in saldf.columns:
                    saldf[diff_col] = saldf[col1] - saldf[col2]
                    diff_count += 1
            elif depth2 > depth1:
                diff_col = f'{col2}-{col1}'
                if diff_col not in saldf.columns:
                    saldf[diff_col] = saldf[col2] - saldf[col1]
                    diff_count += 1

expected_diff_count = len(base_columns) * (len(base_columns) - 1) // 2
print(f"Added {diff_count} unique difference columns")
print(f"Final DataFrame shape: {saldf.shape}")

# Generate the desired order dynamically
available_columns = saldf.columns.tolist()
desired_order = generate_column_order(available_columns)
print(f"Plot order ({len(desired_order)} plots):")
for i, col in enumerate(desired_order):
    print(f"  {i+1:2d}: {col}")

# Global source for all selected points
all_points_source = ColumnDataSource(data=dict(x=[], y=[], plot=[]))

# Global dictionary to store ColumnDataSources for each plot
plot_sources = {}

def create_bokeh_plot(df, column_name, plot_index):
    # Remove NaN values for plotting
    clean_data = df[column_name].dropna()
    if len(clean_data) == 0:
        print(f"Warning: No valid data for column {column_name}")
        return None

    source = ColumnDataSource(data=dict(x=clean_data.index, y=clean_data.values))
    selected_source = ColumnDataSource(data=dict(x=[], y=[]))
    line_source = ColumnDataSource(data=dict(x=[], y=[]))

    p = figure(title=column_name, x_axis_type="datetime", width=1200, height=300)

    # Add light grey background
    p.background_fill_color = "#f5f5f5"
    p.background_fill_alpha = 0.5

    # Add dotted green line at y=0 spanning entire x-range
    p.add_layout(Span(location=0, dimension='width', line_color='green',
                     line_dash='dotted', line_width=1))

    # Original data line
    main_line = p.line('x', 'y', source=source)

    # Create hover tool and add to plot
    hover = HoverTool(
        tooltips=[("Date", "@x{%F}"), ("Value", "@y")],
        formatters={"@x": "datetime"},
        renderers=[main_line],
        anchor="right",
        attachment="left"
    )
    p.add_tools(hover)

    # Store plot sources
    plot_sources[plot_index] = {
        'selected': selected_source,
        'line': line_source,
        'plot': p
    }

    # Selected points with smaller circles and no hover
    p.scatter('x', 'y', source=selected_source, size=5, color='red', fill_color='red')

    # Dotted line connecting selected points
    p.line('x', 'y', source=line_source, line_dash='dotted', line_width=2, color='red')

    def callback(event):
        new_x, new_y = event.x, event.y
        selected_source.stream(dict(x=[new_x], y=[new_y]))
        all_points_source.stream(dict(x=[new_x], y=[new_y], plot=[plot_index]))

        x_data = selected_source.data['x']
        y_data = selected_source.data['y']
        sorted_indices = sorted(range(len(x_data)), key=lambda k: x_data[k])
        line_source.data = dict(x=[x_data[i] for i in sorted_indices],
                              y=[y_data[i] for i in sorted_indices])

    p.on_event('tap', callback)

    return p

def modify_doc(doc):
    plots = []
    plot_count = 0

    for i, column_name in enumerate(desired_order):
        if column_name in saldf.columns:
            plot = create_bokeh_plot(saldf, column_name, i)
            if plot is not None:
                plots.append(plot)
                plot_count += 1

    if not plots:
        print("Error: No plots could be created!")
        return

    print(f"Created {len(plots)} interactive plots from {len(desired_order)} difference columns")

    def save_points():
        df = pd.DataFrame({
            'x': all_points_source.data['x'],
            'y': all_points_source.data['y'],
            'plot': all_points_source.data['plot']
        })

        if len(df) == 0:
            print("No points to save!")
            return

        df['DT'] = pd.to_datetime(df['x'], unit='ms')
        final = df[['DT', 'y', 'plot']].sort_values('DT')

        output_filename = f'adj{mooring_segment}.inputs'

        # Get date range from the data
        start_date = saldf.index.min()
        end_date = saldf.index.max()

        # Create depth list and column mapping based on sorted base columns
        depths_list = []
        col_to_number = {}

        # Sort base columns by depth for consistent ordering
        sorted_base_columns = sorted(base_columns, key=get_depth_number)

        for i, col in enumerate(sorted_base_columns):
            # Map column name to output column number (1-based)
            col_to_number[col] = i + 1

            # Get the depth value for the depths line
            if col == 'SSS':
                depths_list.append(1)
            else:
                depth_num = int(col.replace('S', ''))
                depths_list.append(depth_num)

        # Get the column number for the inst_col specified on command line
        if inst_col in col_to_number:
            inst_col_num = col_to_number[inst_col]
        else:
            print(f"Warning: inst_col '{inst_col}' not found in column mapping, using 1")
            inst_col_num = 1

        # Check if we have enough points to create adjustment pairs
        if len(final) < 2:
            print("Need at least 2 points to create adjustment pairs!")
            return

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(output_filename)
        write_mode = 'a' if file_exists else 'w'

        try:
            with open(output_filename, write_mode) as f:
                # Only write headers if this is a new file
                if not file_exists:
                    # Write header information
                    f.write(f"start = {start_date:%Y%j%H%M%S}\n")
                    f.write(f"end = {end_date:%Y%j%H%M%S}\n")

                    # Write depths header
                    depths_str = "depths: " + "".join(f"{d:6d}" for d in depths_list)
                    f.write(f"{depths_str}\n")
                    f.write("\n")

                    # Write column numbers header
                    col_numbers = "".join(f"{i:7d}" for i in range(1, len(sorted_base_columns) + 1))
                    f.write(f"        {col_numbers}\n")

                    # Write field descriptions
                    f.write("start - stop startoff->stopoff field startzero - stopzero\n")
                    f.write("#Adjust to CTD data where possible:\n")
                else:
                    # Add separator comment for new entries when appending
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"#New entries added on {current_time}:\n")

                # Write data points - ALL points use the same inst_col_num from command line
                for i in range(len(final) - 1):
                    line = f"{final.iloc[i]['DT']:%Y%j%H%M%S} {final.iloc[i+1]['DT']:%Y%j%H%M%S} " \
                           f"{final.iloc[i]['y']:8.4f} {final.iloc[i+1]['y']:8.4f} {inst_col_num}\n"
                    f.write(line)

            if file_exists:
                print(f"Data appended to existing {output_filename}")
            else:
                print(f"New file created: {output_filename}")

            print(f"Number of adjustment points saved: {len(final) - 1}")
            print(f"Date range: {start_date:%Y%j%H%M%S} to {end_date:%Y%j%H%M%S}")
            print(f"Depths: {depths_list}")
            print(f"Column mapping: {dict(sorted(col_to_number.items(), key=lambda x: x[1]))}")
            print(f"Using inst_col '{inst_col}' → column number {inst_col_num}")

        except Exception as e:
            print(f"ERROR writing file: {str(e)}")
            import traceback
            traceback.print_exc()

    def clear_points():
        all_points_source.data = dict(x=[], y=[], plot=[])
        for sources in plot_sources.values():
            sources['selected'].data = dict(x=[], y=[])
            sources['line'].data = dict(x=[], y=[])
        print("All points cleared")

    def clear_last_point():
        # Check if there are any points to remove
        if len(all_points_source.data['x']) == 0:
            print("No points to remove")
            return

        # Get the last point info
        last_plot_index = all_points_source.data['plot'][-1]

        # Remove last point from global source
        new_data = {}
        for key in all_points_source.data:
            new_data[key] = all_points_source.data[key][:-1]
        all_points_source.data = new_data

        # Remove last point from the specific plot's selected source
        if last_plot_index in plot_sources:
            selected_source = plot_sources[last_plot_index]['selected']
            line_source = plot_sources[last_plot_index]['line']

            # Remove last point from selected source
            new_selected_data = {}
            for key in selected_source.data:
                new_selected_data[key] = selected_source.data[key][:-1]
            selected_source.data = new_selected_data

            # Update the connecting line
            x_data = selected_source.data['x']
            y_data = selected_source.data['y']
            if len(x_data) > 0:
                sorted_indices = sorted(range(len(x_data)), key=lambda k: x_data[k])
                line_source.data = dict(x=[x_data[i] for i in sorted_indices],
                                      y=[y_data[i] for i in sorted_indices])
            else:
                line_source.data = dict(x=[], y=[])

        print("Last point cleared")

    def exit_program():
        print("Exiting the program...")
        sys.exit(0)



    # Create buttons
    save_button_top = Button(label="Save Points", button_type="success")
    save_button_top.on_click(save_points)

    clear_button_top = Button(label="Clear All Points", button_type="warning")
    clear_button_top.on_click(clear_points)

    clear_last_button_top = Button(label="Clear Last Point", button_type="warning")
    clear_last_button_top.on_click(clear_last_point)

    exit_button_top = Button(label="Exit", button_type="danger")
    exit_button_top.on_click(exit_program)



    save_button_bottom = Button(label="Save Points", button_type="success")
    save_button_bottom.on_click(save_points)

    clear_button_bottom = Button(label="Clear All Points", button_type="warning")
    clear_button_bottom.on_click(clear_points)

    clear_last_button_bottom = Button(label="Clear Last Point", button_type="warning")
    clear_last_button_bottom.on_click(clear_last_point)

    exit_button_bottom = Button(label="Exit", button_type="danger")
    exit_button_bottom.on_click(exit_program)

    # Create file name display
    file_name_div = Div(text=f"<h1 style='margin: 0; color: #333; text-align: center; font-size: 24px;'>{os.path.basename(dep_file)}</h1>",
                        styles={'width': '100%', 'text-align': 'center'})

    # Create button rows
    top_button_row = row(save_button_top, clear_button_top, clear_last_button_top, exit_button_top)
    bottom_button_row = row(save_button_bottom, clear_button_bottom, clear_last_button_bottom, exit_button_bottom)

    # Create layout
    layout = bokeh_column([file_name_div, top_button_row] + plots + [bottom_button_row])
    doc.add_root(layout)

# Set up the Bokeh server
server = Server({'/': Application(FunctionHandler(modify_doc))})
server.start()

print('Opening Bokeh application on http://localhost:5006/')

server.io_loop.add_callback(server.show, "/")
server.io_loop.start()
