import pandas as pd
import numpy as np
import itertools

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

def profile_noc(df):
    df_noc_send = df[df["  zone name"] == "TEST-NoC-sender-notile"]
    df_noc_send = df_noc_send[[" core_x", " core_y", " time[cycles since reset]"]]
    df_noc_send_group = df_noc_send.groupby([" core_x", " core_y"])
    df_noc_send_cycle = df_noc_send_group.apply(max_min_diff)
    print(df_noc_send_cycle)

profile_noc(df)