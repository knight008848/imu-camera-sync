"""Visualization: timestamp distribution, sync quality checks."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_timestamps(imu_data: dict, camera_data: dict, show: bool = True):
    """Plot IMU and camera timestamp distributions for inspection."""
    imu_ts = imu_data["timestamps"]
    cam_ts = camera_data["timestamps"]

    imu_intervals_ms = np.diff(imu_ts) * 1000
    cam_intervals_ms = np.diff(cam_ts) * 1000

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(imu_intervals_ms, bins=50, kde=True, ax=axes[0])
    median_imu_ms = np.median(imu_intervals_ms)
    axes[0].axvline(median_imu_ms, color="red", linestyle="--", label=f"median={median_imu_ms:.2f}ms")
    axes[0].set_title("IMU Sample Intervals")
    axes[0].set_xlabel("Interval (ms)")
    axes[0].legend()

    sns.histplot(cam_intervals_ms, bins=50, kde=True, ax=axes[1])
    median_cam_ms = np.median(cam_intervals_ms)
    axes[1].axvline(median_cam_ms, color="red", linestyle="--", label=f"median={median_cam_ms:.2f}ms")
    axes[1].set_title("Camera Frame Intervals")
    axes[1].set_xlabel("Interval (ms)")
    axes[1].legend()

    # Normalize to relative time for overlay
    imu_rel = imu_ts - imu_ts[0]
    cam_rel = cam_ts - cam_ts[0]
    axes[2].eventplot(imu_rel, lineoffsets=1, colors=["C0"], label="IMU")
    axes[2].eventplot(cam_rel, lineoffsets=0, colors=["C1"], label="Camera")
    axes[2].set_title("Sensor Timeline Overlay")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Camera", "IMU"])
    axes[2].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_sync_quality(imu_data: dict, camera_data: dict, synced: dict, show: bool = True):
    """Plot before/after alignment comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    imu_ts = imu_data["timestamps"]

    if "imu_indices" in synced and synced["imu_indices"] is not None:
        deltas_ms = (imu_ts[synced["imu_indices"]] - synced["timestamps"]) * 1000
        sns.histplot(deltas_ms, bins=50, kde=True, ax=axes[0])
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].set_title("Alignment Error (Nearest-Neighbor)")
        axes[0].set_xlabel("Timestamp delta (ms)")
    else:
        axes[0].text(0.5, 0.5, "No imu_indices (interp mode)", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Alignment Error (N/A)")

    imu_rel = imu_ts - imu_ts[0]
    synced_rel = synced["timestamps"] - synced["timestamps"][0]
    sns.scatterplot(x=imu_rel[::10], y=imu_data["accel"][::10, 0], s=4, alpha=0.5, label="IMU (orig)", ax=axes[1])
    sns.scatterplot(x=synced_rel, y=synced["accel_aligned"][:, 0], s=4, alpha=0.5, label="Aligned", ax=axes[1])
    axes[1].set_title("Accel X: Before vs Aligned")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    if show:
        plt.show()
    return fig
