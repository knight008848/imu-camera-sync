"""Data loading: IMU (CSV/JSON) and Camera (video/image sequences)."""

from pathlib import Path


def load_imu(path: str | Path):
    """
    Load IMU data from an iPhone sensor log file.

    Supports CSV and JSON formats. Returns a dict with keys:
    - timestamps: 1D array (seconds)
    - accel: Nx3 array (m/s²)
    - gyro: Nx3 array (rad/s)
    """
    raise NotImplementedError


def load_camera(path: str | Path):
    """
    Load camera data from a video file or image sequence directory.

    Returns a dict with keys:
    - timestamps: 1D array (seconds)
    - fps: float
    - frame_count: int
    """
    raise NotImplementedError
