import tempfile
from pathlib import Path

import numpy as np

from imu_camera_sync import loader


def test_load_imu():
    csv_content = (
        "timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n"
        "0.0,0.1,0.2,0.3,0.01,0.02,0.03\n"
        "0.01,0.11,0.21,0.31,0.011,0.021,0.031\n"
        "0.02,0.12,0.22,0.32,0.012,0.022,0.032\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        data = loader.load_imu(tmp_path)
        assert data["timestamps"].shape == (3,)
        assert data["accel"].shape == (3, 3)
        assert data["gyro"].shape == (3, 3)
        np.testing.assert_array_almost_equal(data["accel"][0], [0.1, 0.2, 0.3])
    finally:
        Path(tmp_path).unlink()


def test_load_camera_synthetic_ts():
    # Create a minimal 2-frame mp4 with OpenCV
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, 30.0, (64, 64))
        for _ in range(5):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        data = loader.load_camera(tmp_path)
        assert data["frame_count"] == 5
        assert data["fps"] == 30.0
        assert data["timestamps"].shape == (5,)
        np.testing.assert_almost_equal(data["timestamps"][1] - data["timestamps"][0], 1 / 30.0)
    finally:
        Path(tmp_path).unlink()
