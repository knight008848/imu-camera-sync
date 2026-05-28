"""Data loading: IMU (CSV) and Camera (video/image sequences)."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_imu(path: str | Path):
    """
    Load IMU data from an iPhone sensor log CSV.

    Returns a dict with keys:
    - timestamps: 1D array (seconds)
    - accel: Nx3 array (m/s²)
    - gyro: Nx3 array (rad/s)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    timestamps = df["timestamp"].to_numpy(dtype=np.float64)

    accel = df[["a_x", "a_y", "a_z"]].to_numpy(dtype=np.float64)

    gyro_cols = ["alpha_x", "alpha_y", "alpha_z"]
    gyro = df[gyro_cols].to_numpy(dtype=np.float64)

    return {
        "timestamps": timestamps,
        "accel": accel,
        "gyro": gyro,
    }


def load_camera(path: str | Path, odometry_path: str | Path | None = None):
    """
    Load camera data from a video file.

    If odometry_path is provided (CSV with per-frame timestamps), those
    absolute timestamps are used. Otherwise synthetic timestamps starting
    from 0 are generated from the video FPS.

    Returns a dict with keys:
    - timestamps: 1D array (seconds)
    - fps: float
    - frame_count: int
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if odometry_path is not None:
        odom = pd.read_csv(odometry_path)
        odom.columns = [c.strip() for c in odom.columns]
        odom_ts = odom["timestamp"].to_numpy(dtype=np.float64)

        if len(odom_ts) < frame_count:
            # Pad with extrapolated timestamps
            expected_interval = np.median(np.diff(odom_ts))
            extra = np.arange(len(odom_ts), frame_count)
            odom_ts = np.concatenate([
                odom_ts,
                odom_ts[-1] + (extra - len(odom_ts) + 1) * expected_interval,
            ])
        timestamps = odom_ts[:frame_count]
    else:
        timestamps = np.arange(frame_count, dtype=np.float64) / fps

    return {
        "timestamps": timestamps,
        "fps": fps,
        "frame_count": frame_count,
    }
