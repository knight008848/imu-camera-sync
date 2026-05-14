"""Time synchronization: interpolation, nearest-neighbor, resampling."""


def align(imu_data: dict, camera_data: dict, method: str = "nearest"):
    """
    Align IMU and camera data to a common timeline.

    Parameters
    ----------
    method : str
        'nearest' — nearest-neighbor matching
        'interp'  — linear interpolation

    Returns aligned data dict.
    """
    raise NotImplementedError
