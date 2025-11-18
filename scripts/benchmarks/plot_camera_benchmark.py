# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false
"""Plot camera sensor benchmark results from CSV file.

This script loads benchmark results from a CSV file and generates visualizations.
Useful for quickly regenerating plots without re-running expensive benchmarks.

Examples:

  - Plot results from a CSV file:
      uv run python scripts/benchmarks/plot_camera_benchmark.py \
        outputs/benchmarks/camera_sensor_depth_with_baseline_720x1280.csv

  - Specify custom output path:
      uv run python scripts/benchmarks/plot_camera_benchmark.py \
        outputs/benchmarks/results.csv \
        --output my_plot.png

  - Choose visualization style:
      uv run python scripts/benchmarks/plot_camera_benchmark.py \
        outputs/benchmarks/results.csv \
        --style grouped_bar
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tyro


def format_large_number(x: float, pos: int | None = None) -> str:
  """Format large numbers in a readable way (e.g., 250K, 2.5M)."""
  if x >= 1_000_000:
    return f"{x / 1_000_000:.1f}M"
  elif x >= 1_000:
    return f"{x / 1_000:.0f}K"
  else:
    return f"{x:.0f}"


def plot_grouped_bars(
  df: pd.DataFrame,
  has_baseline: bool,
  device_name: str,
  figsize: tuple[float, float] = (16, 6),
) -> tuple:
  """Create grouped bar chart visualization.

  Args:
    df: DataFrame with benchmark results
    has_baseline: Whether baseline data is included
    device_name: Name of the device used for benchmarking
    figsize: Figure size in inches (width, height)

  Returns:
    Tuple of (fig, axes)
  """
  # Prepare data - convert to throughput (env-steps per second)
  df_plot = df.copy()
  df_plot["throughput"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_mean"]
  df_plot["throughput_min"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_max"]
  df_plot["throughput_max"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_min"]

  # Get unique values
  num_envs_list = sorted(df_plot["num_envs"].unique())

  if has_baseline:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: FPS comparison with baseline
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

    # Plot baseline
    baseline_throughput = []
    baseline_err = []
    for num_env in num_envs_list:
      row = df_baseline[df_baseline["num_envs"] == num_env].iloc[0]
      baseline_throughput.append(row["throughput"])
      baseline_err.append(
        [
          row["throughput"] - row["throughput_min"],
          row["throughput_max"] - row["throughput"],
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
            throughput_vals.append(row["throughput"])
            throughput_err.append(
              [
                row["throughput"] - row["throughput_min"],
                row["throughput_max"] - row["throughput"],
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
    ax1.set_ylabel("Throughput (env-steps/sec)", fontsize=12, fontweight="bold")
    ax1.set_title("Throughput Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(num_envs_list)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_large_number))
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right plot: Overhead percentage as grouped bars
    if "overhead_pct" in df.columns:
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
    # Single plot without baseline
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
            throughput_vals.append(row["throughput"])
            throughput_err.append(
              [
                row["throughput"] - row["throughput_min"],
                row["throughput_max"] - row["throughput"],
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
    ax1.set_ylabel("Throughput (env-steps/sec)", fontsize=12, fontweight="bold")
    ax1.set_title("Camera Sensor Throughput", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(num_envs_list)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_large_number))
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Device: {device_name}", fontsize=11, y=0.98)

  return fig, (ax1, ax2) if has_baseline else (ax1,)


def plot_heatmap(
  df: pd.DataFrame,
  has_baseline: bool,
  device_name: str,
  figsize: tuple[float, float] = (14, 6),
) -> tuple:
  """Create heatmap visualization.

  Args:
    df: DataFrame with benchmark results
    has_baseline: Whether baseline data is included
    device_name: Name of the device used for benchmarking
    figsize: Figure size in inches (width, height)

  Returns:
    Tuple of (fig, axes)
  """

  # Prepare data
  df_plot = df.copy()
  df_plot["throughput"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_mean"]

  # Separate camera data (exclude baseline)
  df_camera = df_plot[df_plot["camera_type"] != "baseline"]

  if df_camera.empty:
    # Fall back to grouped bars if no camera data
    return plot_grouped_bars(df, has_baseline, device_name, figsize)

  camera_types = sorted(df_camera["camera_type"].unique())

  if has_baseline:
    # Create figure with subplots for each camera type + overhead
    n_plots = len(camera_types) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
      axes = [axes]

    # Plot heatmap for each camera type
    for idx, camera_type in enumerate(camera_types):
      df_cam = df_camera[df_camera["camera_type"] == camera_type]

      # Pivot data for heatmap
      pivot_throughput = df_cam.pivot_table(
        index="resolution", columns="num_envs", values="throughput"
      )

      im = axes[idx].imshow(pivot_throughput.values, cmap="RdYlGn", aspect="auto")

      # Set ticks and labels
      axes[idx].set_xticks(np.arange(len(pivot_throughput.columns)))
      axes[idx].set_yticks(np.arange(len(pivot_throughput.index)))
      axes[idx].set_xticklabels(pivot_throughput.columns)
      axes[idx].set_yticklabels(pivot_throughput.index)

      # Add text annotations
      for i in range(len(pivot_throughput.index)):
        for j in range(len(pivot_throughput.columns)):
          axes[idx].text(
            j,
            i,
            f"{pivot_throughput.values[i, j]:.0f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            fontweight="bold",
          )

      axes[idx].set_xlabel("Number of Environments", fontsize=11, fontweight="bold")
      axes[idx].set_ylabel("Resolution", fontsize=11, fontweight="bold")
      axes[idx].set_title(
        f"{camera_type.upper()} Throughput", fontsize=12, fontweight="bold"
      )

      # Add colorbar
      cbar = plt.colorbar(im, ax=axes[idx])
      cbar.set_label("Throughput (env-steps/sec)", fontsize=10)

    # Plot overhead heatmap
    if "overhead_pct" in df.columns and len(camera_types) > 0:
      df_overhead = df_camera.copy()

      # Take first camera type for overhead visualization
      df_ovh = df_overhead[df_overhead["camera_type"] == camera_types[0]]
      pivot_overhead = df_ovh.pivot_table(
        index="resolution", columns="num_envs", values="overhead_pct"
      )

      im = axes[-1].imshow(pivot_overhead.values, cmap="Reds", aspect="auto")

      axes[-1].set_xticks(np.arange(len(pivot_overhead.columns)))
      axes[-1].set_yticks(np.arange(len(pivot_overhead.index)))
      axes[-1].set_xticklabels(pivot_overhead.columns)
      axes[-1].set_yticklabels(pivot_overhead.index)

      for i in range(len(pivot_overhead.index)):
        for j in range(len(pivot_overhead.columns)):
          axes[-1].text(
            j,
            i,
            f"{pivot_overhead.values[i, j]:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            fontweight="bold",
          )

      axes[-1].set_xlabel("Number of Environments", fontsize=11, fontweight="bold")
      axes[-1].set_ylabel("Resolution", fontsize=11, fontweight="bold")
      axes[-1].set_title("Overhead vs Baseline", fontsize=12, fontweight="bold")

      cbar = plt.colorbar(im, ax=axes[-1])
      cbar.set_label("Overhead (%)", fontsize=10)

    fig.suptitle(f"Device: {device_name}", fontsize=12, y=0.98)

  else:
    # Single heatmap for each camera type
    n_plots = len(camera_types)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
      axes = [axes]

    for idx, camera_type in enumerate(camera_types):
      df_cam = df_camera[df_camera["camera_type"] == camera_type]

      pivot_throughput = df_cam.pivot_table(
        index="resolution", columns="num_envs", values="throughput"
      )

      im = axes[idx].imshow(pivot_throughput.values, cmap="RdYlGn", aspect="auto")

      axes[idx].set_xticks(np.arange(len(pivot_throughput.columns)))
      axes[idx].set_yticks(np.arange(len(pivot_throughput.index)))
      axes[idx].set_xticklabels(pivot_throughput.columns)
      axes[idx].set_yticklabels(pivot_throughput.index)

      for i in range(len(pivot_throughput.index)):
        for j in range(len(pivot_throughput.columns)):
          axes[idx].text(
            j,
            i,
            f"{pivot_throughput.values[i, j]:.0f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            fontweight="bold",
          )

      axes[idx].set_xlabel("Number of Environments", fontsize=11, fontweight="bold")
      axes[idx].set_ylabel("Resolution", fontsize=11, fontweight="bold")
      axes[idx].set_title(
        f"{camera_type.upper()} Throughput", fontsize=12, fontweight="bold"
      )

      cbar = plt.colorbar(im, ax=axes[idx])
      cbar.set_label("Throughput (env-steps/sec)", fontsize=10)

    fig.suptitle(f"Device: {device_name}", fontsize=12, y=0.98)

  return fig, axes


def plot_faceted(
  df: pd.DataFrame,
  has_baseline: bool,
  device_name: str,
  figsize: tuple[float, float] = (16, 10),
) -> tuple:
  """Create faceted subplot visualization.

  Args:
    df: DataFrame with benchmark results
    has_baseline: Whether baseline data is included
    device_name: Name of the device used for benchmarking
    figsize: Figure size in inches (width, height)

  Returns:
    Tuple of (fig, axes)
  """
  df_plot = df.copy()
  df_plot["throughput"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_mean"]
  df_plot["throughput_min"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_max"]
  df_plot["throughput_max"] = df_plot["num_envs"] * 1000.0 / df_plot["per_step_ms_min"]

  df_camera = df_plot[df_plot["camera_type"] != "baseline"]

  if df_camera.empty:
    return plot_grouped_bars(df, has_baseline, device_name, figsize)

  resolutions = sorted(df_camera["resolution"].unique())
  camera_types = sorted(df_camera["camera_type"].unique())

  # Create grid of subplots (one per resolution)
  n_res = len(resolutions)
  n_cols = min(3, n_res)
  n_rows = (n_res + n_cols - 1) // n_cols

  fig, axes_array = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
  axes = axes_array.flatten()

  for idx, resolution in enumerate(resolutions):
    ax = axes[idx]  # type: ignore[index]
    df_res = df_camera[df_camera["resolution"] == resolution]

    for camera_type in camera_types:
      df_cam = df_res[df_res["camera_type"] == camera_type].sort_values("num_envs")

      if not df_cam.empty:
        ax.errorbar(  # type: ignore[union-attr]
          df_cam["num_envs"],
          df_cam["throughput"],
          yerr=[
            df_cam["throughput"] - df_cam["throughput_min"],
            df_cam["throughput_max"] - df_cam["throughput"],
          ],
          marker="o",
          linewidth=2.5,
          markersize=8,
          label=camera_type,
          capsize=6,
          capthick=2,
        )

    # Add baseline if available
    if has_baseline:
      df_baseline = df_plot[df_plot["camera_type"] == "baseline"].sort_values(
        "num_envs"
      )
      if not df_baseline.empty:
        ax.errorbar(  # type: ignore[union-attr]
          df_baseline["num_envs"],
          df_baseline["throughput"],
          yerr=[
            df_baseline["throughput"] - df_baseline["throughput_min"],
            df_baseline["throughput_max"] - df_baseline["throughput"],
          ],
          marker="s",
          linewidth=2.5,
          markersize=8,
          label="Baseline",
          linestyle="--",
          capsize=6,
          capthick=2,
          alpha=0.7,
        )

    ax.set_xlabel("Number of Environments", fontsize=10, fontweight="bold")  # type: ignore[union-attr]
    ax.set_ylabel("Throughput (env-steps/sec)", fontsize=10, fontweight="bold")  # type: ignore[union-attr]
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_large_number))  # type: ignore[union-attr]
    ax.set_title(f"Resolution: {resolution}", fontsize=11, fontweight="bold")  # type: ignore[union-attr]
    ax.legend(fontsize=9)  # type: ignore[union-attr]
    ax.grid(True, alpha=0.3)  # type: ignore[union-attr]

  # Hide unused subplots
  for idx in range(n_res, len(axes)):
    axes[idx].axis("off")  # type: ignore[union-attr]

  fig.suptitle(f"Device: {device_name}", fontsize=13, fontweight="bold", y=0.99)
  plt.tight_layout()

  return fig, axes


def generate_plot(
  csv_path: Path,
  output_path: Path | None = None,
  style: Literal["grouped_bar", "heatmap", "faceted", "all"] = "grouped_bar",
  figsize: tuple[float, float] = (16, 6),
) -> None:
  """Generate visualization from benchmark CSV data.

  Args:
    csv_path: Path to CSV file with benchmark results
    output_path: Path for output PNG (default: same as CSV with .png extension)
    style: Visualization style to use
    figsize: Figure size in inches (width, height)
  """
  # Load data
  df = pd.read_csv(csv_path)

  # Check if baseline is included
  has_baseline = "baseline" in df["camera_type"].values

  # Get device name from first row
  device_name = df.iloc[0]["device"]

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

  if style == "all":
    # Generate all visualization types
    base_path = csv_path.with_suffix("")

    # Grouped bar chart
    fig, _ = plot_grouped_bars(df, has_baseline, device_name, figsize)
    out_path = Path(str(base_path) + "_grouped_bar.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grouped bar plot saved to: {out_path}")

    # Heatmap
    fig, _ = plot_heatmap(df, has_baseline, device_name, (14, 6))
    out_path = Path(str(base_path) + "_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to: {out_path}")

    # Faceted
    fig, _ = plot_faceted(df, has_baseline, device_name, (16, 10))
    out_path = Path(str(base_path) + "_faceted.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Faceted plot saved to: {out_path}")

  else:
    # Determine output path
    if output_path is None:
      output_path = csv_path.with_suffix(".png")

    # Generate selected visualization
    if style == "grouped_bar":
      fig, _ = plot_grouped_bars(df, has_baseline, device_name, figsize)
    elif style == "heatmap":
      fig, _ = plot_heatmap(df, has_baseline, device_name, figsize)
    elif style == "faceted":
      fig, _ = plot_faceted(df, has_baseline, device_name, figsize)
    else:
      raise ValueError(f"Unknown style: {style}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def main(
  csv_path: Path,
  output: Path | None = None,
  style: Literal["grouped_bar", "heatmap", "faceted", "all"] = "grouped_bar",
  width: float = 16,
  height: float = 6,
) -> None:
  """Plot camera sensor benchmark results from CSV.

  Args:
    csv_path: Path to CSV file with benchmark results
    output: Output PNG path (default: same as CSV with .png extension)
    style: Visualization style (grouped_bar, heatmap, faceted, or all)
    width: Figure width in inches
    height: Figure height in inches
  """
  if not csv_path.exists():
    print(f"Error: CSV file not found: {csv_path}")
    sys.exit(1)

  generate_plot(csv_path, output, style=style, figsize=(width, height))


if __name__ == "__main__":
  tyro.cli(main)
