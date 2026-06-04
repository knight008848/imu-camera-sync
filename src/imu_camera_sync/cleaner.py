"""Data cleaning: outlier removal, gap filling, timestamp repair."""

import numpy as np
from scipy import interpolate


def clean_imu(imu_data: dict) -> dict:
    """
    Clean IMU data: remove outliers, fill gaps, validate timestamps.

    Returns cleaned data with a ``timestamp_jumps`` key: list of
    (index, prev_ts, curr_ts, delta_s) for each backward jump > 1e-4 s.
    """
    ts = imu_data["timestamps"].copy()
    accel = imu_data["accel"].copy()
    gyro = imu_data["gyro"].copy()

    accel = _clean_signal(accel)
    gyro = _clean_signal(gyro)
    ts, timestamp_jumps = _repair_timestamps(ts)

    return {
        "timestamps": ts,
        "accel": accel,
        "gyro": gyro,
        "timestamp_jumps": timestamp_jumps,
    }


def clean_camera(camera_data: dict) -> dict:
    """
    Clean camera data: validate frame timestamps, detect dropped frames.

    Returns cleaned data with a ``dropped_gaps`` key: list of
    (frame_index, interval_s, expected_s, delta_s) for each gap.
    """
    ts = camera_data["timestamps"].copy()
    intervals_s = np.diff(ts)
    expected_s = np.median(intervals_s)
    gaps = intervals_s > 1.5 * expected_s

    dropped_gaps = []
    if np.any(gaps):
        for i in np.where(gaps)[0]:
            dropped_gaps.append((int(i), float(intervals_s[i]), float(expected_s),
                                 float(intervals_s[i] - expected_s)))

    return {
        "timestamps": ts,
        "fps": camera_data["fps"],
        "frame_count": camera_data["frame_count"],
        "dropped_gaps": dropped_gaps,
    }


def _clean_signal(data: np.ndarray) -> np.ndarray:
    """Detect and interpolate outliers using median absolute deviation.

    Outliers at the edges are clamped to the nearest inlier value
    instead of being linearly extrapolated, which can produce extreme results.
    """
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
            f = interpolate.interp1d(
                x_in, y_in, kind="linear",
                bounds_error=False,
                fill_value=(float(y_in[0]), float(y_in[-1])),
            )
            cleaned[outliers, axis] = f(np.arange(n)[outliers])

    return cleaned


def _repair_timestamps(ts: np.ndarray) -> tuple[np.ndarray, list]:
    """Ensure timestamps are strictly monotonic and non-negative delta.

    Backward jumps larger than 1e-4 s are recorded in the returned list
    as (index, prev_ts, curr_ts, delta_s).
    """
    repaired = ts.copy()
    jumps = []
    for i in range(1, len(repaired)):
        delta_s = repaired[i] - repaired[i - 1]
        if delta_s < -1e-4:
            jumps.append((i, float(repaired[i - 1]), float(repaired[i]), float(delta_s)))
        if delta_s <= 0:
            repaired[i] = repaired[i - 1] + 1e-9
    return repaired, jumps
