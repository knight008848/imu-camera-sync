import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

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


def test_load_imu_missing_timestamp_col():
    csv_content = "time, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n0.0,0.1,0.2,0.3,0.01,0.02,0.03\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="timestamp"):
            loader.load_imu(tmp_path)
    finally:
        Path(tmp_path).unlink()


def test_load_imu_missing_accel_col():
    csv_content = "timestamp, a_x, a_z, alpha_x, alpha_y, alpha_z\n0.0,0.1,0.2,0.01,0.02,0.03\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="a_y"):
            loader.load_imu(tmp_path)
    finally:
        Path(tmp_path).unlink()


def test_load_imu_missing_gyro_col():
    csv_content = "timestamp, a_x, a_y, a_z, alpha_x, alpha_z\n0.0,0.1,0.2,0.3,0.01,0.02\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="alpha_y"):
            loader.load_imu(tmp_path)
    finally:
        Path(tmp_path).unlink()


def test_load_camera_synthetic_ts():
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
        assert data["creation_utc"] is None  # temp file has no creation_time
        np.testing.assert_almost_equal(data["timestamps"][1] - data["timestamps"][0], 1 / 30.0)
    finally:
        Path(tmp_path).unlink()


def test_load_camera_with_creation_time():
    """When MP4 has creation_time, timestamps should be UTC-based."""
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, 30.0, (64, 64))
        for _ in range(3):
            writer.write(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        writer.release()

        # Pretend ffprobe found a creation_time
        utc = 1779776250.0
        with mock.patch.object(loader, "_extract_creation_time", return_value=utc):
            data = loader.load_camera(tmp_path)

        assert data["creation_utc"] == utc
        assert data["timestamps"][0] == utc  # first frame at creation_time
        assert data["timestamps"][1] == pytest.approx(utc + 1 / 30.0)
        assert data["timestamps"][-1] == pytest.approx(utc + 2 / 30.0)
    finally:
        Path(tmp_path).unlink()


# ── boundary / error-path tests ──────────────────────────────────────────


def _mock_capture(fps: float, frame_count: int):
    """Return a mock that behaves like an opened VideoCapture with given props."""

    def _get(prop):
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(frame_count)
        return 0.0

    return mock.MagicMock(isOpened=mock.MagicMock(return_value=True), get=_get)


def test_load_camera_fps_zero():
    with mock.patch("imu_camera_sync.loader.cv2.VideoCapture") as vc:
        vc.return_value = _mock_capture(0.0, 100)
        with pytest.raises(ValueError, match="FPS"):
            loader.load_camera("fake.mp4")


def test_load_camera_fps_negative():
    with mock.patch("imu_camera_sync.loader.cv2.VideoCapture") as vc:
        vc.return_value = _mock_capture(-1.0, 100)
        with pytest.raises(ValueError, match="FPS"):
            loader.load_camera("fake.mp4")


def test_load_camera_fps_nan():
    with mock.patch("imu_camera_sync.loader.cv2.VideoCapture") as vc:
        vc.return_value = _mock_capture(float("nan"), 100)
        with pytest.raises(ValueError, match="FPS"):
            loader.load_camera("fake.mp4")


def test_load_camera_frame_count_zero():
    with mock.patch("imu_camera_sync.loader.cv2.VideoCapture") as vc:
        vc.return_value = _mock_capture(30.0, 0)
        with pytest.raises(ValueError, match="frame count"):
            loader.load_camera("fake.mp4")


def test_load_camera_odometry_missing_timestamp_col():
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf:
        video_path = vf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as of:
        of.write("frame,x,y,z\n0,1,2,3\n1,4,5,6\n")
        odom_path = of.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (64, 64))
        writer.write(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        writer.write(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        writer.release()

        with pytest.raises(ValueError, match="timestamp"):
            loader.load_camera(video_path, odometry_path=odom_path)
    finally:
        Path(video_path).unlink()
        Path(odom_path).unlink()


def test_load_camera_odometry_single_row():
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf:
        video_path = vf.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as of:
        of.write("timestamp,x,y,z\n0.0,1,2,3\n")
        odom_path = of.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (64, 64))
        for _ in range(5):
            writer.write(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        writer.release()

        with pytest.raises(ValueError, match="need at least 2"):
            loader.load_camera(video_path, odometry_path=odom_path)
    finally:
        Path(video_path).unlink()
        Path(odom_path).unlink()
