"""Example: load, clean, and synchronize IMU+Camera data."""

from imu_camera_sync import cleaner, loader, synchronizer, visualizer

DATA_DIR = "data/79c1787d6c"
IMU_PATH = f"{DATA_DIR}/imu.csv"
CAMERA_PATH = f"{DATA_DIR}/rgb.mp4"
ODOMETRY_PATH = f"{DATA_DIR}/odometry.csv"


def main():
    # 1. Load
    imu_data = loader.load_imu(IMU_PATH)
    camera_data = loader.load_camera(CAMERA_PATH, odometry_path=ODOMETRY_PATH)

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
