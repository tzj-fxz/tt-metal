import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def moe_plot():
    # Read the Excel file
    data = pd.read_excel("MoE-Random-K.xlsx")
    
    # Get unique values for Core_x and Core_y
    core_x_values = data['Core x'].unique()
    core_y_values = data['Core y'].unique()
    
    # Create a plot for each combination of Core_x and Core_y
    for cx in core_x_values:
        for cy in core_y_values:
            # Filter data for current core configuration
            core_data = data[(data['Core x'] == cx) & (data['Core y'] == cy)]
            
            if len(core_data) == 0:
                continue
                
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Get unique M/core values
            m_per_core_values = core_data['M/core'].unique()
            
            # Plot a line for each M/core value
            for m in m_per_core_values:
                m_data = core_data[core_data['M/core'] == m]
                plt.plot(m_data['K/core'], m_data['Bandwidth(GB/s)'], marker='o', label=f'M/core={m}')
            
            plt.xlabel('K/core')
            plt.ylabel('Bandwidth (GB/s)')
            plt.title(f'Bandwidth vs K/core for Core ({cx}, {cy})')
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            plt.savefig(f'bandwidth_plot_core_{cx}_{cy}.png')
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, choices=['moe'], default='moe')
    args = parser.parse_args()
    if args.mode == "moe":
        moe_plot()


if __name__ == "__main__":
    main()
