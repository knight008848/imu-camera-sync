import numpy as np

from imu_camera_sync import cleaner


def test_clean_imu_outlier_detection():
    ts = np.arange(0, 1.0, 0.01)
    n = len(ts)
    accel = np.zeros((n, 3))
    accel[:, 0] = np.sin(ts * 10)
    accel[:, 1] = np.cos(ts * 10)

    # Inject known outliers
    accel[10, 0] = 100.0  # far outlier
    accel[50, 0] = -50.0  # far outlier

    data = {"timestamps": ts, "accel": accel, "gyro": np.zeros((n, 3))}
    result = cleaner.clean_imu(data)

    # Outlier values should be replaced (not extreme anymore)
    assert abs(result["accel"][10, 0]) < 10.0
    assert abs(result["accel"][50, 0]) < 10.0
    # Non-outlier index should be unchanged
    np.testing.assert_almost_equal(result["accel"][20, 0], accel[20, 0])


def test_clean_imu_timestamps():
    ts = np.array([0.0, 0.01, 0.02, 0.02, 0.03])  # duplicate at idx 3
    accel = np.zeros((5, 3))
    gyro = np.zeros((5, 3))
    data = {"timestamps": ts, "accel": accel, "gyro": gyro}

    result = cleaner.clean_imu(data)
    assert np.all(np.diff(result["timestamps"]) > 0)


def test_clean_camera_no_gaps():
    cam_data = {
        "timestamps": np.arange(100) / 30.0,
        "fps": 30.0,
        "frame_count": 100,
    }
    result = cleaner.clean_camera(cam_data)
    assert result["frame_count"] == 100
    assert result["fps"] == 30.0


def test_clean_camera_dropped_frames_detected(capsys):
    ts = np.arange(100) / 30.0
    # Introduce a gap
    ts[50:] += 0.5

    cam_data = {"timestamps": ts, "fps": 30.0, "frame_count": 100}
    result = cleaner.clean_camera(cam_data)
    captured = capsys.readouterr()
    assert "dropped-frame gap" in captured.out
    assert result["frame_count"] == 100
