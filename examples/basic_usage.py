"""Example: load, clean, and synchronize IMU+Camera data."""

from imu_camera_sync import loader, cleaner, synchronizer, visualizer

# TODO: replace with actual data paths
IMU_PATH = "data/imu_sample.csv"
CAMERA_PATH = "data/video_sample.mp4"


def main():
    # 1. Load
    imu_data = loader.load_imu(IMU_PATH)
    camera_data = loader.load_camera(CAMERA_PATH)

    # 2. Inspect timestamps
    visualizer.plot_timestamps(imu_data, camera_data)

    # 3. Clean
    imu_clean = cleaner.clean_imu(imu_data)
    camera_clean = cleaner.clean_camera(camera_data)

    # 4. Align
    synced = synchronizer.align(imu_clean, camera_clean, method="nearest")

    # 5. Check result
    visualizer.plot_sync_quality(imu_data, camera_data, synced)


if __name__ == "__main__":
    main()
