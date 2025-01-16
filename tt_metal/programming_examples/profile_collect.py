import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

def max_min_diff(group):
    return group.max() - group.min()

file_path = "/home/zhengju.tang/tt-metal/generated/profiler/.logs/profile_log_device.csv"

df = pd.read_csv(file_path, skiprows=1)
print(df.columns)

max_time = max(df[" time[cycles since reset]"])
min_time = min(df[" time[cycles since reset]"])
print("max time", max_time)
print("min time", min_time)
print("whole time", max_time-min_time)

def profile_cannon(df):
    # df_bmm_shift_unpack = df_bmm_shift[df[" RISC processor type"] == "TRISC_0"]
    # df_bmm_shift_unpack_begin = df_bmm_shift_unpack[df[" zone phase"] == "begin"].reset_index()
    # df_bmm_shift_math = df_bmm_shift[df[" RISC processor type"] == "TRISC_2"]
    # df_bmm_shift_math_end = df_bmm_shift_math[df[" zone phase"] == "end"].reset_index()
    # result = df_bmm_shift_math_end[" time[cycles since reset]"] - df_bmm_shift_unpack_begin[" time[cycles since reset]"]
    # print(result)
    # print("len result: ", len(result))
    # print(result.mean())
    df_core = df[[" core_x", " core_y"]]
    df_core = df_core.drop_duplicates()
    df_core_repeated = df_core.loc[np.repeat(df_core.index, 6)].reset_index(drop=True)
    print(df_core_repeated)

    df_reader_all = df[df["  zone name"] == "TEST-reader_bmm_cannon_shift"]
    df_reader_group = df_reader_all.groupby([" core_x", " core_y"])
    df_reader_begin = df_reader_all[df_reader_all[" zone phase"] == "begin"].reset_index()
    df_reader_end = df_reader_all[df_reader_all[" zone phase"] == "end"].reset_index()
    result = df_reader_end[" time[cycles since reset]"] - df_reader_begin[" time[cycles since reset]"]

    # in place modify
    df_core_repeated["cycles"] = result
    print(df_core_repeated)
    df_core_repeated.to_csv("output_reader_shift.csv", index=False)

    df_reader_shift = df[df["  zone name"] == "TEST-reader_bmm_cannon_shift"]
    grouped_df_reader_shift = df_reader_shift.groupby([" core_x", " core_y", " RISC processor type", " zone phase"])
    df_cycles = grouped_df_reader_shift[" time[cycles since reset]"].mean().rename("cycles")
    grouped_df_bmm_risc = df_cycles.groupby([" core_x", " core_y", " RISC processor type"])
    df_diff = grouped_df_bmm_risc.apply(max_min_diff)
    df_diff.to_csv("output_shift.csv", index=False)
    print(df_diff)
    max_cycles = max(df_diff)
    print("reader shift cycles", max_cycles)

def profile_cannon_fig(df):
    # Convert cycles to milliseconds
    df['time_ms'] = df[' time[cycles since reset]']

    # Create a unique identifier for each core
    # df['core_id'] = 'Core(' + df[' core_x'].astype(str) + ',' + df[' core_y'].astype(str) + ')'
    df['core_id'] = 'Core_x:' + df[' core_x'].astype(str) + ', Core_y:' + df[' core_y'].astype(str)

    # Create figure
    plt.figure(figsize=(15, 20))

    # Create a timeline plot
    for processor in ['NCRISC']:
        processor_data = df[df[' RISC processor type'] == processor]
        
        # Get unique zones for this processor
        zones = processor_data['  zone name'].unique()
        zones = [zone for zone in zones if zone.startswith("TEST-reader_bmm_cannon")]

        # Create subplot
        plt.subplot(1, 1, 1 if processor == 'BRISC' else 1)
        
        # Plot each zone's begin and end times
        for i, core in enumerate(sorted(processor_data['core_id'].unique())):
            core_data = processor_data[processor_data['core_id'] == core]
            # Calculate offset for each zone
            zone_offsets = {zone: idx * 0.2 for idx, zone in enumerate(zones)}
            for zone in zones:
                zone_data = core_data[core_data['  zone name'] == zone]
                begins = zone_data[zone_data[' zone phase'] == 'begin']['time_ms']
                ends = zone_data[zone_data[' zone phase'] == 'end']['time_ms']
                colors = plt.cm.rainbow(np.linspace(0, 1, len(begins)))
                
                if not begins.empty and not ends.empty:
                    for t, (begin, end) in enumerate(zip(begins, ends)):
                        y_pos = i + zone_offsets[zone]
                        plt.hlines(y=y_pos, xmin=begin, xmax=end, 
                                label=zone if i == 0 else "",
                                color=colors[t], 
                                linewidth=8, alpha=0.5)
        
        plt.yticks(range(len(processor_data['core_id'].unique())), 
                sorted(processor_data['core_id'].unique()))
        plt.title(f'{processor} Processor Timeline')
        plt.xlabel('Cycle')
        plt.ylabel('Core ID')
        plt.grid(True, alpha=0.3)
        if processor == 'NCRISC':
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig("noc.png")

def profile_noc(df):
    df_noc_send = df[df["  zone name"] == "TEST-NoC-sender-bandwidth"]
    df_noc_send = df_noc_send[[" core_x", " core_y", " time[cycles since reset]"]]
    df_noc_send_group = df_noc_send.groupby([" core_x", " core_y"])
    df_noc_send_cycle = df_noc_send_group.apply(max_min_diff)
    print(df_noc_send_cycle)

def profile_noc_warmup(df):
    df_noc_send = df[df["  zone name"] == "TEST-NoC-sender-warmup"]
    df_noc_send = df_noc_send[[" core_x", " core_y", " time[cycles since reset]"]]
    df_noc_send_group = df_noc_send.groupby([" core_x", " core_y"])
    df_noc_send_cycle = df_noc_send_group.apply(max_min_diff)
    print(df_noc_send_cycle)

def profile_noc_dram(df):
    df_noc_send = df[df["  zone name"] == "TEST-NoC-sender_dram"]
    df_noc_send = df_noc_send[[" core_x", " core_y", " time[cycles since reset]"]]
    df_noc_send_group = df_noc_send.groupby([" core_x", " core_y"])
    df_noc_send_cycle = df_noc_send_group.apply(max_min_diff)
    print(df_noc_send_cycle)

def profile_noc_fig(df):
    # Convert cycles to milliseconds
    df['time_ms'] = df[' time[cycles since reset]']

    # Create a unique identifier for each core
    # df['core_id'] = 'Core(' + df[' core_x'].astype(str) + ',' + df[' core_y'].astype(str) + ')'
    df['core_id'] = 'Core_x:' + df[' core_x'].astype(str) + ', Core_y:' + df[' core_y'].astype(str)

    # Create figure
    plt.figure(figsize=(15, 10))

    # Create a timeline plot
    for processor in ['BRISC']:
        processor_data = df[df[' RISC processor type'] == processor]
        
        # Get unique zones for this processor
        zones = processor_data['  zone name'].unique()
        zones = [zone for zone in zones if zone.startswith("TEST")]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(zones)))
        zone_colors = dict(zip(zones, colors))

        # Create subplot
        plt.subplot(1, 1, 1 if processor == 'BRISC' else 1)
        
        # Plot each zone's begin and end times
        for i, core in enumerate(sorted(processor_data['core_id'].unique())):
            core_data = processor_data[processor_data['core_id'] == core]
            # Calculate offset for each zone
            zone_offsets = {zone: idx * 0.2 for idx, zone in enumerate(zones)}
            for zone in zones:
                zone_data = core_data[core_data['  zone name'] == zone]
                begins = zone_data[zone_data[' zone phase'] == 'begin']['time_ms']
                ends = zone_data[zone_data[' zone phase'] == 'end']['time_ms']
                
                if not begins.empty and not ends.empty:
                    y_pos = i + zone_offsets[zone]
                    plt.hlines(y=y_pos, xmin=begins, xmax=ends, 
                            label=zone if i == 0 else "",
                            color=zone_colors[zone], 
                            linewidth=8, alpha=0.5)
        
        plt.yticks(range(len(processor_data['core_id'].unique())), 
                sorted(processor_data['core_id'].unique()))
        plt.title(f'{processor} Processor Timeline')
        plt.xlabel('Cycle')
        plt.ylabel('Core ID')
        plt.grid(True, alpha=0.3)
        if processor == 'BRISC':
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig("noc.png")

if __name__ == "__main__":
    profile_cannon(df)
    profile_cannon_fig(df)
    # profile_noc_dram(df)
    # profile_noc_warmup(df)
    # profile_noc(df)
    # profile_noc_fig(df)