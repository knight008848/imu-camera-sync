"""Example: pipeline single session with visualization."""

from imu_camera_sync import pipeline

SESSION_DIR = "data/79c1787d6c"


def main():
    result = pipeline.process_session(
        SESSION_DIR, method="nearest", with_visualization=True
    )

    print(f"Session: {result['session']}")
    print(f"IMU:     {result['imu']}")
    print(f"Camera:  {result['camera']}")
    print(f"Align:   {result['alignment']}")
    print(f"Output:  {result['output']}")


if __name__ == "__main__":
    main()
