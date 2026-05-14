"""Data cleaning: outlier removal, gap filling, timestamp repair."""


def clean_imu(imu_data: dict):
    """
    Clean IMU data: remove outliers, fill gaps, validate timestamps.

    Returns cleaned data in the same format.
    """
    raise NotImplementedError


def clean_camera(camera_data: dict):
    """
    Clean camera data: validate frame timestamps, detect dropped frames.

    Returns cleaned data in the same format.
    """
    raise NotImplementedError
