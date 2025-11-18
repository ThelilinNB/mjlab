"""Compare before/after benchmark results to show optimization impact."""

import sys
from pathlib import Path

import pandas as pd


def compare_benchmarks(before_csv: Path, after_csv: Path):
  """Compare two benchmark CSV files and show the improvement."""

  df_before = pd.read_csv(before_csv)
  df_after = pd.read_csv(after_csv)

  print("=" * 80)
  print("RENDERING OPTIMIZATION RESULTS")
  print("=" * 80)
  print()

  # Merge on configuration columns
  merge_cols = ["num_envs", "resolution", "camera_type"]
  df_merged = df_before.merge(df_after, on=merge_cols, suffixes=("_before", "_after"))

  # Calculate speedup
  df_merged["speedup"] = (
    df_merged["per_step_ms_mean_before"] / df_merged["per_step_ms_mean_after"]
  )
  df_merged["improvement_pct"] = (
    (df_merged["per_step_ms_mean_before"] - df_merged["per_step_ms_mean_after"])
    / df_merged["per_step_ms_mean_before"]
    * 100
  )

  # Calculate physics throughput (with decimation=4)
  decimation = 4
  df_merged["throughput_before"] = (
    df_merged["num_envs"] * 1000.0 / df_merged["per_step_ms_mean_before"] * decimation
  )
  df_merged["throughput_after"] = (
    df_merged["num_envs"] * 1000.0 / df_merged["per_step_ms_mean_after"] * decimation
  )

  # Print results table
  print(
    f"{'Config':<25} {'Before (ms)':<15} {'After (ms)':<15} {'Speedup':<10} {'Improvement':<12}"
  )
  print("-" * 80)

  for _, row in df_merged.iterrows():
    config = f"{row['num_envs']} envs @ {row['resolution']}"
    before_ms = f"{row['per_step_ms_mean_before']:.2f}"
    after_ms = f"{row['per_step_ms_mean_after']:.2f}"
    speedup = f"{row['speedup']:.2f}x"
    improvement = f"{row['improvement_pct']:.1f}%"

    print(
      f"{config:<25} {before_ms:<15} {after_ms:<15} {speedup:<10} {improvement:<12}"
    )

  print()
  print("=" * 80)
  print("THROUGHPUT COMPARISON (physics-steps/sec)")
  print("=" * 80)
  print()

  print(f"{'Config':<25} {'Before':<15} {'After':<15} {'Increase':<12}")
  print("-" * 80)

  for _, row in df_merged.iterrows():
    config = f"{row['num_envs']} envs @ {row['resolution']}"
    before_thr = f"{row['throughput_before'] / 1000:.1f}K"
    after_thr = f"{row['throughput_after'] / 1000:.1f}K"
    increase = f"{row['speedup']:.2f}x"

    print(f"{config:<25} {before_thr:<15} {after_thr:<15} {increase:<12}")

  print()
  print("=" * 80)
  print("SUMMARY STATISTICS")
  print("=" * 80)
  print()

  # Camera-only rows (exclude baseline)
  df_camera = df_merged[df_merged["camera_type"] != "baseline"]

  avg_speedup = df_camera["speedup"].mean()
  min_speedup = df_camera["speedup"].min()
  max_speedup = df_camera["speedup"].max()
  avg_improvement = df_camera["improvement_pct"].mean()

  print(f"Average speedup: {avg_speedup:.2f}x ({avg_improvement:.1f}% faster)")
  print(f"Min speedup: {min_speedup:.2f}x")
  print(f"Max speedup: {max_speedup:.2f}x")
  print()

  # Check if overhead scaling improved
  print("OVERHEAD SCALING ANALYSIS:")
  print("-" * 80)

  for resolution in df_camera["resolution"].unique():
    df_res = df_camera[df_camera["resolution"] == resolution]
    if len(df_res) >= 2:
      # Compare overhead at different env counts
      df_res_sorted = df_res.sort_values("num_envs")
      first_row = df_res_sorted.iloc[0]
      last_row = df_res_sorted.iloc[-1]

      overhead_before_first = first_row["overhead_pct_before"]
      overhead_before_last = last_row["overhead_pct_before"]
      overhead_after_first = first_row["overhead_pct_after"]
      overhead_after_last = last_row["overhead_pct_after"]

      scaling_before = overhead_before_last / overhead_before_first
      scaling_after = overhead_after_last / overhead_after_first

      print(f"\n{resolution}:")
      print(
        f"  Before: {first_row['num_envs']} envs = {overhead_before_first:.1f}% OH, "
        f"{last_row['num_envs']} envs = {overhead_before_last:.1f}% OH "
        f"(scales {scaling_before:.2f}x)"
      )
      print(
        f"  After:  {first_row['num_envs']} envs = {overhead_after_first:.1f}% OH, "
        f"{last_row['num_envs']} envs = {overhead_after_last:.1f}% OH "
        f"(scales {scaling_after:.2f}x)"
      )
      print(
        f"  Improvement: {(scaling_before - scaling_after) / scaling_before * 100:.1f}% better scaling"
      )


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python compare_results.py <before.csv> <after.csv>")
    sys.exit(1)

  before_csv = Path(sys.argv[1])
  after_csv = Path(sys.argv[2])

  if not before_csv.exists():
    print(f"Error: {before_csv} not found")
    sys.exit(1)
  if not after_csv.exists():
    print(f"Error: {after_csv} not found")
    sys.exit(1)

  compare_benchmarks(before_csv, after_csv)
