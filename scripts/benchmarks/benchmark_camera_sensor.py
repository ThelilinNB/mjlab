"""Benchmark throughput of the camera sensor.

This script benchmarks per-step time while varying:
- camera type: RGB, depth
- image resolutions (height x width)
- number of environments

The benchmark uses the tracking environment configuration from demo.py.

Examples:

  - Benchmark both RGB and depth cameras across resolutions:
      uv run python scripts/benchmarks/benchmark_camera_sensor.py \\
        --num_envs 256 512 \\
        --resolutions 240x320,480x640 --steps 200 --warmup 20

  - Only depth camera at 720p:
      uv run python scripts/benchmarks/benchmark_camera_sensor.py \\
        --num_envs 256 --resolutions 720x1280 --camera_type depth
"""

from __future__ import annotations

import os

os.environ["MJLAB_WARP_QUIET"] = "1"
import contextlib
import io
import sys
import time
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import tyro
from tqdm import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.gcs import ensure_default_motion
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg


@dataclass
class BenchmarkConfig:
  """Configuration for camera sensor benchmarking."""

  num_envs: tuple[int, ...] = (256, 512)
  """List of environment counts to benchmark."""

  resolutions: str = "240x320,480x640"
  """Comma-separated list of HxW resolutions (e.g., 240x320,480x640)."""

  camera_type: Literal["rgb", "depth", "both"] = "both"
  """Camera type to benchmark: rgb, depth, or both."""

  include_baseline: bool = True
  """Include baseline benchmark without camera sensors for comparison."""

  plot_overhead: bool = False
  """Include overhead plot in visualization (only applies when include_baseline=True)."""

  num_runs: int = 1
  """Number of runs per configuration to measure variance."""

  steps: int = 200
  """Steps per run to time."""

  warmup: int = 20
  """Warmup steps per run before timing."""

  device: str = "cuda:0"
  """Device to run benchmark on."""


@contextlib.contextmanager
def suppress_output():
  """Suppress stdout and stderr output."""
  old_stdout = sys.stdout
  old_stderr = sys.stderr
  try:
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    yield
  finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr


def parse_resolutions(res_str: str) -> list[tuple[int, int]]:
  """Parse resolution string into list of (height, width) tuples."""
  resolutions: list[tuple[int, int]] = []
  for token in [s for s in res_str.split(",") if s]:
    h, w = token.lower().split("x")
    resolutions.append((int(h), int(w)))
  print(f"[INFO]: Resolutions: {resolutions}")
  return resolutions


def create_env_with_camera(
  num_envs: int,
  height: int,
  width: int,
  camera_type: Literal["rgb", "depth"] | None,
  device: str,
) -> tuple[ManagerBasedRlEnv, int]:
  """Create tracking environment with optional camera sensor.

  Args:
    num_envs: Number of parallel environments
    height: Camera image height (ignored if camera_type is None)
    width: Camera image width (ignored if camera_type is None)
    camera_type: Type of camera sensor ("rgb", "depth", or None for baseline)
    device: Device to run on

  Returns:
    Tuple of (configured environment, decimation)
  """
  # Get base tracking env config (use play=True for simpler setup)
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=True)

  # Override num_envs
  env_cfg.scene.num_envs = num_envs

  # Set motion file to the default demo motion
  if env_cfg.commands is not None:
    motion_path = ensure_default_motion()
    for cmd_cfg in env_cfg.commands.values():
      if isinstance(cmd_cfg, MotionCommandCfg):
        cmd_cfg.motion_file = motion_path

  # Remove all camera sensors from original config
  # Keep only the contact sensor
  sensors = tuple(
    s for s in env_cfg.scene.sensors if not isinstance(s, CameraSensorCfg)
  )

  # Add camera sensor if requested (otherwise baseline without camera)
  if camera_type is not None:
    dt = 1.0 / 20.0  # 20 Hz update rate
    # Use robot/depth camera for both rgb and depth (just change the type)
    camera_cfg = CameraSensorCfg(
      name=f"{camera_type}_camera",
      camera_name="robot/depth",
      width=width,
      height=height,
      type=(camera_type,),
      update_period=dt,
    )
    sensors = sensors + (camera_cfg,)

  env_cfg.scene.sensors = sensors

  # Get decimation from environment config
  decimation = env_cfg.decimation

  # Create environment (suppress verbose output)
  with suppress_output():
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return env, decimation


def run_benchmark(
  env: ManagerBasedRlEnv,
  steps: int,
  warmup: int,
  device: str,
) -> dict[str, int | float | str]:
  """Run benchmark and return timing results."""
  action_dim = env.single_action_space.shape[0]
  # Warmup
  for _ in tqdm(range(warmup), desc="Warmup", leave=False):
    env.step(torch.zeros((env.num_envs, action_dim), device=device))

  # Get initial memory
  if device.startswith("cuda"):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

  # Timing
  t0 = time.perf_counter()
  for _ in tqdm(range(steps), desc="Benchmark", leave=False):
    env.step(torch.zeros((env.num_envs, action_dim), device=device))

  if device.startswith("cuda"):
    torch.cuda.synchronize(device)
  t1 = time.perf_counter()

  per_step_ms = (t1 - t0) / steps * 1000

  # Memory usage
  avg_memory = 0.0
  if device.startswith("cuda"):
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # Convert to MB
    avg_memory = peak_memory

  return {
    "per_step_ms": float(per_step_ms),
    "avg_memory_mb": float(avg_memory),
  }


def main(config: BenchmarkConfig) -> None:
  """Run camera sensor benchmarks."""
  print("=" * 80)
  print("Camera Sensor Benchmark")
  print("=" * 80)

  resolutions = parse_resolutions(config.resolutions)

  # Determine which camera types to benchmark
  camera_types: list[Literal["rgb", "depth"]] = []
  if config.camera_type == "both":
    camera_types = ["rgb", "depth"]
  else:
    camera_types = [config.camera_type]  # type: ignore

  device_name = (
    torch.cuda.get_device_name(torch.cuda.current_device())
    if torch.cuda.is_available()
    else "CPU"
  )

  results: list[dict[str, int | float | str]] = []

  # Build list of configurations to benchmark
  configs_to_run: list[tuple[int, tuple[int, int] | None, str | None]] = []

  # Add baseline runs (no camera) if requested
  if config.include_baseline:
    for num_envs in config.num_envs:
      configs_to_run.append((num_envs, None, None))

  # Add camera runs
  for num_envs in config.num_envs:
    for resolution in resolutions:
      for camera_type in camera_types:
        configs_to_run.append((num_envs, resolution, camera_type))

  with tqdm(total=len(configs_to_run), desc="Overall Progress") as pbar:
    for num_envs, resolution, camera_type in configs_to_run:
      # Format description
      if camera_type is None:
        desc = f"Envs: {num_envs}, Baseline (no camera)"
        height, width = 0, 0  # Unused for baseline
        resolution_str = "baseline"
        camera_type_str = "baseline"
      else:
        assert resolution is not None
        height, width = resolution
        desc = f"Envs: {num_envs}, Res: {height}x{width}, Type: {camera_type}"
        resolution_str = f"{height}x{width}"
        camera_type_str = camera_type

      pbar.set_description(desc)

      # Run multiple times to measure variance
      run_results = []
      decimation = None
      for _ in range(config.num_runs):
        # Create environment
        env, env_decimation = create_env_with_camera(
          num_envs=num_envs,
          height=height,
          width=width,
          camera_type=camera_type,  # type: ignore
          device=config.device,
        )
        if decimation is None:
          decimation = env_decimation

        # Run benchmark
        result = run_benchmark(
          env=env,
          steps=config.steps,
          warmup=config.warmup,
          device=config.device,
        )
        run_results.append(result)

        # Clean up
        env.close()
        del env
        if config.device.startswith("cuda"):
          torch.cuda.empty_cache()

      # Aggregate results across runs
      per_step_times = [r["per_step_ms"] for r in run_results]
      memory_vals = [r["avg_memory_mb"] for r in run_results]

      aggregated_result = {
        "num_envs": num_envs,
        "resolution": resolution_str,
        "camera_type": camera_type_str,
        "device": device_name,
        "decimation": decimation,
        "per_step_ms_mean": float(np.mean(per_step_times)),
        "per_step_ms_std": float(np.std(per_step_times)),
        "per_step_ms_min": float(np.min(per_step_times)),
        "per_step_ms_max": float(np.max(per_step_times)),
        "avg_memory_mb": float(np.mean(memory_vals)),
        "num_runs": config.num_runs,
      }
      results.append(aggregated_result)

      # Update progress bar with results
      pbar.set_postfix(
        {
          "per_step_ms": f"{aggregated_result['per_step_ms_mean']:.3f}±{aggregated_result['per_step_ms_std']:.3f}",
          "memory_mb": f"{aggregated_result['avg_memory_mb']:.1f}",
        }
      )
      pbar.update(1)

  # Save results
  df = pd.DataFrame(results)

  # Calculate overhead percentage if baseline is included
  if config.include_baseline:
    overhead_rows = []
    for _, row in df.iterrows():
      if row["camera_type"] != "baseline":
        # Find corresponding baseline for same num_envs
        baseline = df[
          (df["num_envs"] == row["num_envs"]) & (df["camera_type"] == "baseline")
        ]
        if not baseline.empty:
          baseline_time = baseline.iloc[0]["per_step_ms_mean"]
          overhead_pct = (row["per_step_ms_mean"] - baseline_time) / baseline_time * 100
          overhead_rows.append(overhead_pct)
        else:
          overhead_rows.append(0.0)
      else:
        overhead_rows.append(0.0)
    df["overhead_pct"] = overhead_rows

  os.makedirs("outputs/benchmarks", exist_ok=True)

  # Create output filename
  camera_suffix = config.camera_type
  if config.include_baseline:
    camera_suffix = f"{camera_suffix}_with_baseline"
  resolution_suffix = config.resolutions.replace(",", "_").replace("x", "x")
  csv_path = f"outputs/benchmarks/camera_sensor_{camera_suffix}_{resolution_suffix}.csv"
  md_path = f"outputs/benchmarks/camera_sensor_{camera_suffix}_{resolution_suffix}.md"

  df.to_csv(csv_path, index=False)
  print(f"\n[INFO]: Results saved to {csv_path}")

  # Create markdown table
  with open(md_path, "w") as f:
    f.write(f"# Camera Sensor Benchmark - {camera_suffix}\n\n")
    f.write(f"**Device**: {device_name}\n\n")
    if config.include_baseline:
      f.write(
        "**Note**: Overhead % shows the slowdown compared to baseline (no camera)\n\n"
      )
    markdown_table = df.to_markdown(index=False, floatfmt=".3f")
    if markdown_table is not None:
      f.write(markdown_table)
    f.write("\n")

  print(f"[INFO]: Markdown table saved to {md_path}")

  # Generate visualization
  plot_path = (
    f"outputs/benchmarks/camera_sensor_{camera_suffix}_{resolution_suffix}.png"
  )
  generate_visualization(df, config, device_name, plot_path, config.plot_overhead)
  print(f"[INFO]: Visualization saved to {plot_path}")


# pyright: reportAttributeAccessIssue=false
def format_large_number(x: float, pos: int | None = None) -> str:
  """Format large numbers in a readable way (e.g., 250K, 2.5M)."""
  if x >= 1_000_000:
    return f"{x / 1_000_000:.1f}M"
  elif x >= 1_000:
    return f"{x / 1_000:.0f}K"
  else:
    return f"{x:.0f}"


def generate_visualization(
  df: pd.DataFrame,
  config: BenchmarkConfig,
  device_name: str,
  output_path: str,
  plot_overhead: bool = False,
) -> None:
  """Generate matplotlib visualization of benchmark results using grouped bar charts."""
  # Set clean style
  plt.rcParams.update(
    {
      "font.size": 11,
      "axes.grid": True,
      "grid.alpha": 0.3,
      "axes.spines.top": False,
      "axes.spines.right": False,
    }
  )

  # Prepare data - convert to throughput (env-steps per second and physics-steps per second)
  df_plot = df.copy()
  df_plot["throughput"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_mean"]
  df_plot["throughput_min"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_max"]
  df_plot["throughput_max"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_min"]

  # Physics throughput (multiply by decimation)
  df_plot["physics_throughput"] = df_plot["throughput"] * df_plot["decimation"]
  df_plot["physics_throughput_min"] = df_plot["throughput_min"] * df_plot["decimation"]
  df_plot["physics_throughput_max"] = df_plot["throughput_max"] * df_plot["decimation"]

  # Get unique values
  num_envs_list = sorted(df_plot["num_envs"].unique())

  if config.include_baseline and plot_overhead:
    # Two subplots: throughput comparison and overhead
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
  else:
    # Single plot: throughput comparison only
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

  if config.include_baseline:
    # Throughput comparison with baseline (grouped bar chart)
    # Separate baseline and camera data
    df_baseline = df_plot[df_plot["camera_type"] == "baseline"]
    df_camera = df_plot[df_plot["camera_type"] != "baseline"]

    # Get unique resolutions (excluding baseline) and sort by numeric value
    resolutions_raw = df_camera["resolution"].unique()
    # Sort resolutions by numeric pixel count (HxW)
    resolutions = sorted(
      resolutions_raw,
      key=lambda r: int(r.split("x")[0]) * int(r.split("x")[1]) if "x" in str(r) else 0,
    )
    camera_types = sorted(df_camera["camera_type"].unique())

    # Set up bar positions
    x = np.arange(len(num_envs_list))
    width = 0.15  # Width of each bar

    # Define consistent color palette for resolutions
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(resolutions)))  # type: ignore[attr-defined]

    # Plot baseline (using physics throughput)
    baseline_throughput = []
    baseline_err = []
    for num_env in num_envs_list:
      row = df_baseline[df_baseline["num_envs"] == num_env].iloc[0]
      baseline_throughput.append(row["physics_throughput"])
      baseline_err.append(
        [
          row["physics_throughput"] - row["physics_throughput_min"],
          row["physics_throughput_max"] - row["physics_throughput"],
        ]
      )

    ax1.bar(
      x - width * len(resolutions) / 2 - width / 2,
      baseline_throughput,
      width,
      label="Baseline (no camera)",
      yerr=np.array(baseline_err).T,
      capsize=5,
      alpha=0.8,
      edgecolor="black",
      linewidth=1.2,
    )

    # Plot each resolution
    for i, resolution in enumerate(resolutions):
      for camera_type in camera_types:
        df_res = df_camera[
          (df_camera["resolution"] == resolution)
          & (df_camera["camera_type"] == camera_type)
        ]

        throughput_vals = []
        throughput_err = []
        for num_env in num_envs_list:
          row = df_res[df_res["num_envs"] == num_env]
          if not row.empty:
            row = row.iloc[0]
            throughput_vals.append(row["physics_throughput"])
            throughput_err.append(
              [
                row["physics_throughput"] - row["physics_throughput_min"],
                row["physics_throughput_max"] - row["physics_throughput"],
              ]
            )
          else:
            throughput_vals.append(0)
            throughput_err.append([0, 0])

        offset = (i - len(resolutions) / 2 + 0.5) * width
        ax1.bar(
          x + offset,
          throughput_vals,
          width,
          label=f"{camera_type} @ {resolution}",
          yerr=np.array(throughput_err).T,
          capsize=5,
          alpha=0.8,
          edgecolor="black",
          linewidth=1.2,
          color=colors[i],
        )

    ax1.set_xlabel("Number of Environments", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Throughput (physics-steps/sec)", fontsize=12, fontweight="bold")
    ax1.set_title("Throughput Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(num_envs_list)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_large_number))
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right plot: Overhead percentage as grouped bars (only if plot_overhead is True)
    if plot_overhead and "overhead_pct" in df.columns:
      df_overhead = df_camera.copy()

      for i, resolution in enumerate(resolutions):
        for camera_type in camera_types:
          df_res = df_overhead[
            (df_overhead["resolution"] == resolution)
            & (df_overhead["camera_type"] == camera_type)
          ]

          overhead_vals = []
          for num_env in num_envs_list:
            row = df_res[df_res["num_envs"] == num_env]
            if not row.empty:
              overhead_vals.append(row.iloc[0]["overhead_pct"])
            else:
              overhead_vals.append(0)

          offset = (i - len(resolutions) / 2 + 0.5) * width
          ax2.bar(
            x + offset,
            overhead_vals,
            width,
            label=f"{camera_type} @ {resolution}",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
            color=colors[i],
          )

      ax2.set_xlabel("Number of Environments", fontsize=12, fontweight="bold")
      ax2.set_ylabel("Overhead (%)", fontsize=12, fontweight="bold")
      ax2.set_title("Camera Overhead vs Baseline", fontsize=14, fontweight="bold")
      ax2.set_xticks(x)
      ax2.set_xticklabels(num_envs_list)
      ax2.legend(fontsize=9, loc="best")
      ax2.grid(True, alpha=0.3, axis="y")
      ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    fig.suptitle(f"Device: {device_name}", fontsize=11, y=0.98)

  else:
    # Single plot: grouped bar chart without baseline
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    df_camera = df_plot.copy()
    # Sort resolutions by numeric pixel count
    resolutions_raw = df_camera["resolution"].unique()
    resolutions = sorted(
      resolutions_raw,
      key=lambda r: int(r.split("x")[0]) * int(r.split("x")[1]) if "x" in str(r) else 0,
    )
    camera_types = sorted(df_camera["camera_type"].unique())

    x = np.arange(len(num_envs_list))
    width = 0.2

    # Define consistent color palette for resolutions
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(resolutions)))  # type: ignore[attr-defined]

    for i, resolution in enumerate(resolutions):
      for camera_type in camera_types:
        df_res = df_camera[
          (df_camera["resolution"] == resolution)
          & (df_camera["camera_type"] == camera_type)
        ]

        throughput_vals = []
        throughput_err = []
        for num_env in num_envs_list:
          row = df_res[df_res["num_envs"] == num_env]
          if not row.empty:
            row = row.iloc[0]
            throughput_vals.append(row["physics_throughput"])
            throughput_err.append(
              [
                row["physics_throughput"] - row["physics_throughput_min"],
                row["physics_throughput_max"] - row["physics_throughput"],
              ]
            )
          else:
            throughput_vals.append(0)
            throughput_err.append([0, 0])

        offset = (i - len(resolutions) / 2 + 0.5) * width
        ax1.bar(
          x + offset,
          throughput_vals,
          width,
          label=f"{camera_type} @ {resolution}",
          yerr=np.array(throughput_err).T,
          capsize=5,
          alpha=0.8,
          edgecolor="black",
          linewidth=1.2,
          color=colors[i],
        )

    ax1.set_xlabel("Number of Environments", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Throughput (physics-steps/sec)", fontsize=12, fontweight="bold")
    ax1.set_title("Camera Sensor Throughput", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(num_envs_list)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_large_number))
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Device: {device_name}", fontsize=11, y=0.98)
  plt.tight_layout()
  plt.savefig(output_path, dpi=150, bbox_inches="tight")
  plt.close()


if __name__ == "__main__":
  config = tyro.cli(BenchmarkConfig)
  main(config)
