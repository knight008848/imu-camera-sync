"""Time synchronization: interpolation, nearest-neighbor, resampling."""

import numpy as np
from scipy import interpolate


def align(imu_data: dict, camera_data: dict, method: str = "nearest") -> dict:
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

    Returns aligned data dict with keys:
    - timestamps: shared camera timestamps
    - accel_aligned: Nx3 array
    - gyro_aligned: Nx3 array
    - imu_indices: indices into the original IMU data
    """
    imu_ts = imu_data["timestamps"]
    cam_ts = camera_data["timestamps"]

    if method == "nearest":
        return _align_nearest(imu_data, imu_ts, cam_ts)
    elif method == "interp":
        return _align_interp(imu_data, imu_ts, cam_ts)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _align_nearest(imu_data: dict, imu_ts: np.ndarray, cam_ts: np.ndarray) -> dict:
    idx = np.searchsorted(imu_ts, cam_ts)
    idx = np.clip(idx, 1, len(imu_ts) - 1)

    left_diff = np.abs(imu_ts[idx - 1] - cam_ts)
    right_diff = np.abs(imu_ts[idx] - cam_ts)
    nearest_idx = np.where(left_diff <= right_diff, idx - 1, idx)

    return {
        "timestamps": cam_ts.copy(),
        "accel_aligned": imu_data["accel"][nearest_idx],
        "gyro_aligned": imu_data["gyro"][nearest_idx],
        "imu_indices": nearest_idx,
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
    }
