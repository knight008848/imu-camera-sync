"""Time synchronization: interpolation, nearest-neighbor, resampling."""

import csv

import numpy as np
from scipy import interpolate

_DEFAULT_TOLERANCE_S = 0.05  # 50 ms — half the typical 100 Hz IMU interval


def align(
    imu_data: dict,
    camera_data: dict,
    method: str = "nearest",
    max_tolerance_s: float = _DEFAULT_TOLERANCE_S,
) -> dict:
    """
    Align IMU and camera data to a common timeline.

    Both data dicts must share the same time base — e.g. both in device
    uptime seconds (from odometry) or both in UTC seconds (converted via
    ``creation_utc`` from the loader).

    Parameters
    ----------
    method : str
        'nearest' — nearest-neighbor matching
        'interp'  — linear interpolation
    max_tolerance_s : float
        Maximum allowed time difference in seconds between a camera frame
        and its matched IMU sample. Frames exceeding this threshold are
        dropped. Only applies to 'nearest' method. Default 0.05 s (50 ms).

    Returns aligned data dict with keys:
    - timestamps: camera timestamps for valid frames
    - accel_aligned: Nx3 array
    - gyro_aligned: Nx3 array
    - imu_indices: indices into the original IMU data
    - dropped: number of frames dropped due to tolerance (0 if none)
    """
    imu_ts = imu_data["timestamps"]
    cam_ts = camera_data["timestamps"]

    if method == "nearest":
        return _align_nearest(imu_data, imu_ts, cam_ts, max_tolerance_s)
    elif method == "interp":
        return _align_interp(imu_data, imu_ts, cam_ts)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _align_nearest(
    imu_data: dict,
    imu_ts: np.ndarray,
    cam_ts: np.ndarray,
    max_tolerance_s: float,
) -> dict:
    idx = np.searchsorted(imu_ts, cam_ts)
    idx = np.clip(idx, 1, len(imu_ts) - 1)

    left_diff_s = np.abs(imu_ts[idx - 1] - cam_ts)
    right_diff_s = np.abs(imu_ts[idx] - cam_ts)
    nearest_idx = np.where(left_diff_s <= right_diff_s, idx - 1, idx)

    time_diff_s = np.abs(imu_ts[nearest_idx] - cam_ts)
    valid_mask = time_diff_s <= max_tolerance_s
    n_dropped = int(np.sum(~valid_mask))

    return {
        "timestamps": cam_ts[valid_mask].copy(),
        "accel_aligned": imu_data["accel"][nearest_idx[valid_mask]],
        "gyro_aligned": imu_data["gyro"][nearest_idx[valid_mask]],
        "imu_indices": nearest_idx[valid_mask],
        "dropped": n_dropped,
    }


def _align_interp(imu_data: dict, imu_ts: np.ndarray, cam_ts: np.ndarray) -> dict:
    accel_interp = interpolate.interp1d(
        imu_ts, imu_data["accel"], axis=0, kind="linear",
        bounds_error=False,
        fill_value=(imu_data["accel"][0], imu_data["accel"][-1]),
    )(cam_ts)

    gyro_interp = interpolate.interp1d(
        imu_ts, imu_data["gyro"], axis=0, kind="linear",
        bounds_error=False,
        fill_value=(imu_data["gyro"][0], imu_data["gyro"][-1]),
    )(cam_ts)

    return {
        "timestamps": cam_ts.copy(),
        "accel_aligned": accel_interp,
        "gyro_aligned": gyro_interp,
        "imu_indices": None,
        "dropped": 0,
    }


def to_csv(synced: dict, path: str) -> None:
    """Export aligned data as a flat CSV table.

    Columns: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    Plus imu_index when available (nearest mode; omitted in interp mode).
    """
    timestamps = synced["timestamps"]
    accel = synced["accel_aligned"]
    gyro = synced["gyro_aligned"]
    imu_indices = synced.get("imu_indices")

    headers = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    if imu_indices is not None:
        headers.append("imu_index")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(len(timestamps)):
            row = [timestamps[i], *accel[i], *gyro[i]]
            if imu_indices is not None:
                row.append(imu_indices[i])
            writer.writerow(row)
