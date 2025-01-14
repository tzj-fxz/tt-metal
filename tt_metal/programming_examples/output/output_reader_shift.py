import pandas as pd
import glob
import os

# Get all CSV files in current directory
# Use os.path to get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = glob.glob(os.path.join(current_dir, 'output_reader_shift*.csv'))

# Initialize empty list to store dataframes
dfs = []

# Read each CSV file and append to list
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    df = df.rename(columns={'cycles': f'cycles_{i+1}'})
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, axis=1)
print(combined_df[:5])

# Find all cycle columns
cycle_columns = [col for col in combined_df.columns if 'cycles' in col]

# Calculate the average of all cycle columns
combined_df['cycle_average'] = combined_df[cycle_columns].mean(axis=1)
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

print("Average cycles per core:")
print(combined_df[[' core_x', ' core_y', 'cycle_average']])

combined_df.to_csv('output_reader_shift.csv')

# Group by core_x and core_y
def calculate_avg_without_max(group):
    # Sort values by cycles_average
    sorted_values = group['cycle_average'].sort_values()
    # Remove the maximum value and calculate mean of remaining values
    return sorted_values[:-1].mean()

# Group by both core coordinates and apply our function
result = combined_df.groupby([' core_x', ' core_y']).agg({
    'cycle_average': lambda x: calculate_avg_without_max(pd.DataFrame({'cycle_average': x}))
}).reset_index()

# Rename the column to be more descriptive
result = result.rename(columns={'cycle_average': 'avg_without_max'})

# Sort by core_x and core_y for better readability
result = result.sort_values([' core_x', ' core_y'])

print("Average cycles (excluding max) for each core:")
print(result)

# Optionally save to a new CSV file
result.to_csv('core_averages_without_max.csv', index=False)
