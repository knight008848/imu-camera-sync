"""Data loading: IMU (CSV) and Camera (video/image sequences)."""

import json
import subprocess
import warnings
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

    required = [
        "timestamp",
        "a_x", "a_y", "a_z",
        "alpha_x", "alpha_y", "alpha_z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"IMU CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

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

    if np.isnan(fps) or fps <= 0:
        raise ValueError(f"Invalid video FPS: {fps}. File may be corrupted.")
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count: {frame_count}. File may be empty or corrupted.")

    creation_utc = None

    if odometry_path is not None:
        odom = pd.read_csv(odometry_path)
        odom.columns = [c.strip() for c in odom.columns]

        if "timestamp" not in odom.columns:
            raise ValueError(
                f"Odometry CSV must contain a 'timestamp' column. "
                f"Found: {list(odom.columns)}"
            )

        odom_ts = odom["timestamp"].to_numpy(dtype=np.float64)

        if len(odom_ts) < frame_count:
            warnings.warn(
                f"Odometry has {len(odom_ts)} timestamps but video has "
                f"{frame_count} frames. Truncating to {len(odom_ts)} frames "
                f"({frame_count - len(odom_ts)} frames dropped)."
            )
            frame_count = len(odom_ts)
        timestamps = odom_ts[:frame_count]
    else:
        creation_utc = _extract_creation_time(path)
        if creation_utc is not None:
            timestamps = creation_utc + np.arange(frame_count, dtype=np.float64) / fps
        else:
            timestamps = np.arange(frame_count, dtype=np.float64) / fps

    return {
        "timestamps": timestamps,
        "fps": fps,
        "frame_count": frame_count,
        "creation_utc": creation_utc,
    }


def _extract_creation_time(path: str | Path) -> float | None:
    """Extract MP4 creation_time as a Unix timestamp via ffprobe.

    Returns None if ffprobe is unavailable or the tag is missing.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        info = json.loads(result.stdout)
        creation_str = info["format"]["tags"]["creation_time"]
        from datetime import datetime

        dt = datetime.fromisoformat(creation_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (FileNotFoundError, KeyError, json.JSONDecodeError,
            subprocess.TimeoutExpired, ValueError):
        return None
