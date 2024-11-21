import pandas as pd
import itertools

def max_min_diff(group):
    return group.max() - group.min()

file_path = "/home/zhengju.tang/tt-metal/generated/profiler/.logs/profile_log_device.csv"

df = pd.read_csv(file_path, skiprows=1)
print(df.columns)

df_bmm = df[df["  zone name"] == "TEST-bmm-start"]
grouped_df_bmm = df_bmm.groupby([" core_x", " core_y", " RISC processor type", " zone phase"])
df_cycles = grouped_df_bmm[" time[cycles since reset]"].mean().rename("cycles")
print(df_cycles)
grouped_df_bmm_risc = df_cycles.groupby([" core_x", " core_y", " RISC processor type"])
df_diff = grouped_df_bmm_risc.apply(max_min_diff)
max_cycles = max(df_diff)
print(max_cycles)
