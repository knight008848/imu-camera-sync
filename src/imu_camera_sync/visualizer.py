"""Visualization: timestamp distribution, sync quality checks."""


def plot_timestamps(imu_data: dict, camera_data: dict):
    """Plot IMU and camera timestamp distributions for inspection."""
    raise NotImplementedError


def plot_sync_quality(imu_data: dict, camera_data: dict, synced: dict):
    """Plot before/after alignment comparison."""
    raise NotImplementedError
