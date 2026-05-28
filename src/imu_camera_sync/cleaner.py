"""Data cleaning: outlier removal, gap filling, timestamp repair."""

import numpy as np
from scipy import interpolate


def clean_imu(imu_data: dict) -> dict:
    """
    Clean IMU data: remove outliers, fill gaps, validate timestamps.

    Returns cleaned data in the same format.
    """
    ts = imu_data["timestamps"].copy()
    accel = imu_data["accel"].copy()
    gyro = imu_data["gyro"].copy()

    accel = _clean_signal(accel)
    gyro = _clean_signal(gyro)
    ts = _repair_timestamps(ts)

    return {"timestamps": ts, "accel": accel, "gyro": gyro}


def clean_camera(camera_data: dict) -> dict:
    """
    Clean camera data: validate frame timestamps, detect dropped frames.

    Returns cleaned data in the same format.
    """
    ts = camera_data["timestamps"].copy()
    intervals = np.diff(ts)
    expected = np.median(intervals)
    gaps = intervals > 1.5 * expected

    if np.any(gaps):
        gap_indices = np.where(gaps)[0]
        for i in gap_indices:
            delta = intervals[i] - expected
            print(f"[clean_camera] dropped-frame gap at frame {i}: "
                  f"interval={intervals[i]:.6f}s (expected ~{expected:.6f}s, delta={delta:.6f}s)")

    return {
        "timestamps": ts,
        "fps": camera_data["fps"],
        "frame_count": camera_data["frame_count"],
    }


def _clean_signal(data: np.ndarray) -> np.ndarray:
    """Detect and interpolate outliers using median absolute deviation."""
    cleaned = data.copy()
    n, n_axes = data.shape

    for axis in range(n_axes):
        col = data[:, axis]
        median = np.median(col)
        mad = np.median(np.abs(col - median))
        if mad == 0:
            continue
        threshold = 5.0 * mad / 0.6745
        outliers = np.abs(col - median) > threshold

        if np.any(outliers):
            inliers = ~outliers
            x_in = np.arange(n)[inliers]
            y_in = col[inliers]
            f = interpolate.interp1d(x_in, y_in, kind="linear", fill_value="extrapolate")
            cleaned[outliers, axis] = f(np.arange(n)[outliers])

    return cleaned


def _repair_timestamps(ts: np.ndarray) -> np.ndarray:
    """Ensure timestamps are strictly monotonic and non-negative delta."""
    repaired = ts.copy()
    for i in range(1, len(repaired)):
        if repaired[i] <= repaired[i - 1]:
            repaired[i] = repaired[i - 1] + 1e-9
    return repaired
