import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import argparse

def max_min_diff(group):
    return group.max() - group.min()

file_path = "/home/zhengju.tang/tt-metal/generated/profiler/.logs/profile_log_device.csv"

df = pd.read_csv(file_path, skiprows=1)
print(df.columns)

def profile_cannon(df):
    # every shift cycle
    df_core = df[[" core_x", " core_y"]]
    df_core = df_core.drop_duplicates()
    df_core_repeated = df_core.loc[np.repeat(df_core.index, 6)].reset_index(drop=True)

    df_reader_all = df[df["  zone name"] == "TEST-reader_bmm_cannon_shift"]
    df_reader_begin = df_reader_all[df_reader_all[" zone phase"] == "begin"].reset_index()
    df_reader_end = df_reader_all[df_reader_all[" zone phase"] == "end"].reset_index()
    df_pack_all = df[df["  zone name"] == "TEST-bmm-shift-pack"]
    df_pack_begin = df_pack_all[df_pack_all[" zone phase"] == "begin"].reset_index()
    df_pack_begin_filter = df_pack_begin[~(df_pack_begin.index % 7 == 6)].reset_index()
    df_pack_end = df_pack_all[df_pack_all[" zone phase"] == "end"].reset_index()
    result = df_reader_end[" time[cycles since reset]"] - df_pack_begin_filter[" time[cycles since reset]"].combine(df_reader_begin[" time[cycles since reset]"], max)

    # average shift cycle
    df_core_repeated["shift-cycles"] = result
    df_core_repeated.to_csv("output_reader_shift.csv", index=False)

    grouped_df_reader_shift = df_core_repeated.groupby([" core_x", " core_y"])
    df_cycles = grouped_df_reader_shift["shift-cycles"].mean()
    df_cycles.to_csv("output_shift.csv", index=False)
    print(df_cycles)
    max_cycles_shift = max(df_cycles)
    avg_cycles_shift = df_cycles.mean()
    
    # every math cycle
    df_core = df[[" core_x", " core_y"]]
    df_core = df_core.drop_duplicates()
    df_core_repeated = df_core.loc[np.repeat(df_core.index, 6)].reset_index(drop=True)

    df_math_all = df[df["  zone name"] == "TEST-bmm-shift-math"]
    df_math_begin = df_math_all[df_math_all[" zone phase"] == "begin"].reset_index()
    df_math_end = df_math_all[df_math_all[" zone phase"] == "end"].reset_index()
    df_unpack_all = df[df["  zone name"] == "TEST-bmm-shift-unpack"]
    df_unpack_begin = df_unpack_all[df_unpack_all[" zone phase"] == "begin"].reset_index()
    df_unpack_end = df_unpack_all[df_unpack_all[" zone phase"] == "end"].reset_index()
    result = df_math_end[" time[cycles since reset]"] - df_unpack_begin[" time[cycles since reset]"].combine(df_math_begin[" time[cycles since reset]"], max)

    # average math cycle
    df_core_repeated["math-cycles"] = result
    print(df_core_repeated)
    df_core_repeated.to_csv("output_math_shift.csv", index=False)

    grouped_df_math_shift = df_core_repeated.groupby([" core_x", " core_y"])
    df_cycles = grouped_df_math_shift["math-cycles"].mean()
    df_cycles.to_csv("output_bmm.csv", index=False)
    print(df_cycles)
    max_cycles_math = max(df_cycles)

    # average bmm cycle
    # df_compute_bmm = df[df["  zone name"] == "TEST-bmm-shift-unpack"]
    # df_compute_bmm = df_compute_bmm[df_compute_bmm[" RISC processor type"] == "TRISC_0"]
    # grouped_df_compute_bmm = df_compute_bmm.groupby([" core_x", " core_y", " RISC processor type", " zone phase"])
    # df_cycles_bmm = grouped_df_compute_bmm[" time[cycles since reset]"].mean().rename("cycles")
    # grouped_df_bmm = df_cycles_bmm.groupby([" core_x", " core_y", " RISC processor type"])
    # df_diff_bmm = grouped_df_bmm.apply(max_min_diff)
    # df_diff_bmm.to_csv("output_bmm.csv", index=False)
    # print(df_diff_bmm)
    # max_cycles_bmm = max(df_diff_bmm)

    # total device cycle
    df_core = df[[" core_x", " core_y"]]
    df_core = df_core.drop_duplicates()
    df_core_repeated = df_core.loc[np.repeat(df_core.index, 6)].reset_index(drop=True)
    df_cannon_begin = df[df["  zone name"] == "TEST-reader_bmm_all"]
    df_cannon_begin = df_cannon_begin[df_cannon_begin[" zone phase"] == "begin"].reset_index()
    df_cannon_end = df[df["  zone name"] == "TEST-writer_bmm_cannon"]
    df_cannon_end = df_cannon_end[df_cannon_end[" zone phase"] == "end"].reset_index()
    result = df_cannon_end[" time[cycles since reset]"] - df_cannon_begin[" time[cycles since reset]"]
    df_core_repeated["total-cycles"] = result
    df_core_repeated.to_csv("output_total.csv", index=False)
    grouped_df_reader_shift = df_core_repeated.groupby([" core_x", " core_y"])
    df_cycles = grouped_df_reader_shift["total-cycles"].max()
    max_cycles_total = max(df_cycles)

    print("reader shift cycles max", max_cycles_shift)
    print("reader shift cycles avg", avg_cycles_shift)
    print("reader bmm cycles", max_cycles_math)
    print("total cannon cycles", max_cycles_total)


def profile_reuse_mcast(df):
    # total device cycle
    df_core = df[[" core_x", " core_y"]]
    df_core = df_core.drop_duplicates()
    df_core_repeated = df_core.loc[np.repeat(df_core.index, 6)].reset_index(drop=True)
    df_cannon_begin = df[df["  zone name"] == "reuse-mcast-reader"]
    df_cannon_begin = df_cannon_begin[df_cannon_begin[" zone phase"] == "begin"].reset_index()
    df_cannon_end = df[df["  zone name"] == "reuse-mcast-writer"]
    df_cannon_end = df_cannon_end[df_cannon_end[" zone phase"] == "end"].reset_index()
    result = df_cannon_end[" time[cycles since reset]"] - df_cannon_begin[" time[cycles since reset]"]
    df_core_repeated["total-cycles"] = result
    df_core_repeated.to_csv("output_total_reuse_mcast.csv", index=False)
    grouped_df_reader_shift = df_core_repeated.groupby([" core_x", " core_y"])
    df_cycles = grouped_df_reader_shift["total-cycles"].max()
    max_cycles_total = max(df_cycles)

    print("total reuse_mcast cycles", max_cycles_total)


def profile_cannon_fig(df):
    # Convert cycles to milliseconds (cycles)
    df['time_cycle'] = df[' time[cycles since reset]']

    # Create a unique identifier for each core
    # df['core_id'] = 'Core(' + df[' core_x'].astype(str) + ',' + df[' core_y'].astype(str) + ')'
    df['core_id'] = 'Core_x:' + df[' core_x'].astype(str) + ', Core_y:' + df[' core_y'].astype(str)

    # Create figure
    plt.figure(figsize=(15, 30))

    # Create a timeline plot
    for processor in ['NCRISC', 'TRISC', 'TRISC', 'TRISC']:
        processor_data = df[df[' RISC processor type'].str.startswith(processor)]
        
        # Get unique zones for this processor
        zones = processor_data['  zone name'].unique()
        # zones = ["TEST-reader_bmm_cannon_initial", "TEST-reader_bmm_cannon_shift", "TEST-bmm-shift"]
        zones = ["TEST-reader_bmm_cannon_shift", "TEST-bmm-shift-unpack", "TEST-bmm-shift-math", "TEST-bmm-shift-pack"]
        zone_color = {
            "TEST-reader_bmm_cannon_shift": 0,
            "TEST-bmm-shift-unpack": 1,
            "TEST-bmm-shift-math": 2,
            "TEST-bmm-shift-pack": 3
        }

        # Create subplot
        plt.subplot(1, 1, 1)
        
        # Plot each zone's begin and end times
        for i, core in enumerate(sorted(processor_data['core_id'].unique())):
            core_data = processor_data[processor_data['core_id'] == core]
            # Calculate offset for each zone
            zone_offsets = {zone: idx * 0.2 for idx, zone in enumerate(zones)}
            for zone in zones:
                zone_data = core_data[core_data['  zone name'] == zone]
                begins = zone_data[zone_data[' zone phase'] == 'begin']['time_cycle']
                ends = zone_data[zone_data[' zone phase'] == 'end']['time_cycle']
                colors = plt.cm.rainbow(np.linspace(0, 1, len(zones) * len(begins)))
                
                if not begins.empty and not ends.empty:
                    for t, (begin, end) in enumerate(zip(begins, ends)):
                        y_pos = i + zone_offsets[zone]
                        xmin = begin
                        if (zone == "TEST-bmm-shift-math" or zone == "TEST-bmm-shift-pack") and t == 0:
                            tmp_data = core_data[core_data['  zone name'] == 'TEST-bmm-shift-unpack']
                            xmin = tmp_data[tmp_data[' zone phase'] == 'begin']['time_cycle']
                            xmin = xmin.iloc[0]
                        plt.hlines(y=y_pos, xmin=xmin, xmax=end, 
                                label=zone if i == 0 else "",
                                color=colors[t * len(zones) + zone_color[zone]], 
                                linewidth=6, alpha=0.5)
        
        plt.yticks(range(len(processor_data['core_id'].unique())), 
                sorted(processor_data['core_id'].unique()))
        plt.title(f'{processor} Processor Timeline')
        plt.xlabel('Cycle')
        plt.ylabel('Core ID')
        plt.grid(True, alpha=0.3)
        # if processor == 'NCRISC':
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig("cannon.png")

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
    df['time_cycle'] = df[' time[cycles since reset]']

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
                begins = zone_data[zone_data[' zone phase'] == 'begin']['time_cycle']
                ends = zone_data[zone_data[' zone phase'] == 'end']['time_cycle']
                
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

def profile_all2all(df):
    df_core = df[[" core_x", " core_y"]]
    df_core = df_core.drop_duplicates().reset_index()
    df_reader_write_to_noc = df[df["  zone name"] == "reader_write_to_noc"]
    df_reader_write_to_noc_begin = df_reader_write_to_noc[df_reader_write_to_noc[" zone phase"] == "begin"].reset_index()
    df_reader_write_to_noc_end = df_reader_write_to_noc[df_reader_write_to_noc[" zone phase"] == "end"].reset_index()
    df_reader_semaphore = df[df["  zone name"] == "reader_semaphore"]
    df_reader_semaphore_begin = df_reader_semaphore[df_reader_semaphore[" zone phase"] == "begin"].reset_index()
    df_reader_semaphore_end = df_reader_semaphore[df_reader_semaphore[" zone phase"] == "end"].reset_index()
    noc_result = df_reader_write_to_noc_end[" time[cycles since reset]"] - df_reader_write_to_noc_begin[" time[cycles since reset]"]
    noc_semaphore_result = df_reader_semaphore_end[" time[cycles since reset]"] - df_reader_semaphore_begin[" time[cycles since reset]"]
    total_result = df_reader_semaphore_end[" time[cycles since reset]"] - df_reader_write_to_noc_begin[" time[cycles since reset]"]
    df_core["all2all-noc-data-cycles"] = noc_result
    df_core["all2all-noc-semaphore-cycles"] = noc_semaphore_result
    df_core["all2all-noc-cycles"] = total_result
    print(df_core)
    df_core.to_csv("output_all2all_noc.csv", index=False)

def profile_moe(df):
    df_core = df[[" core_x", " core_y"]]
    df_core = df_core.drop_duplicates().reset_index()
    df_reader_write_to_noc = df[df["  zone name"] == "reader_random_send"]
    df_reader_write_to_noc_begin = df_reader_write_to_noc[df_reader_write_to_noc[" zone phase"] == "begin"].reset_index()
    df_reader_write_to_noc_end = df_reader_write_to_noc[df_reader_write_to_noc[" zone phase"] == "end"].reset_index()
    df_reader_semaphore = df[df["  zone name"] == "reader_semaphore"]
    df_reader_semaphore_begin = df_reader_semaphore[df_reader_semaphore[" zone phase"] == "begin"].reset_index()
    df_reader_semaphore_end = df_reader_semaphore[df_reader_semaphore[" zone phase"] == "end"].reset_index()
    noc_result = df_reader_write_to_noc_end[" time[cycles since reset]"] - df_reader_write_to_noc_begin[" time[cycles since reset]"]
    noc_semaphore_result = df_reader_semaphore_end[" time[cycles since reset]"] - df_reader_semaphore_begin[" time[cycles since reset]"]
    total_result = df_reader_semaphore_end[" time[cycles since reset]"] - df_reader_write_to_noc_begin[" time[cycles since reset]"]
    df_core["moe-random-send-data-cycles"] = noc_result
    df_core["moe-random-send-semaphore-cycles"] = noc_semaphore_result
    df_core["moe-random-send-all-cycles"] = total_result
    print(df_core)
    df_core.to_csv("output_moe_random_noc.csv", index=False)
    print("max data cycle: ", max(noc_result))
    print("max total cycle: ", max(total_result))

def main():
    parser = argparse.ArgumentParser(description="choose type to profile")
    parser.add_argument('--mode', '-m', type=str, choices=['noc', 'cannon', 'all2all', 'reusemcast', 'moe'], default='moe')
    args = parser.parse_args()
    if args.mode == 'noc':
        profile_noc(df)
        profile_noc_fig(df)
    elif args.mode == 'cannon':
        profile_cannon(df)
        profile_cannon_fig(df)
    elif args.mode == 'all2all':
        profile_all2all(df)
    elif args.mode == 'reusemcast':
        profile_reuse_mcast(df)
    elif args.mode == 'moe':
        profile_moe(df)


if __name__ == "__main__":
    main()
