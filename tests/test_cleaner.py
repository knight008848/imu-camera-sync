import numpy as np
import pytest

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


def test_clean_camera_dropped_frames_detected():
    ts = np.arange(100) / 30.0
    # Introduce a gap
    ts[50:] += 0.5

    cam_data = {"timestamps": ts, "fps": 30.0, "frame_count": 100}
    result = cleaner.clean_camera(cam_data)
    assert len(result["dropped_gaps"]) > 0
    assert result["dropped_gaps"][0][0] == 49  # gap at frame index 49
    assert result["frame_count"] == 100


def test_clean_imu_edge_outlier_clamped():
    """Outlier at index 0 should be clamped to nearest inlier, not extrapolated."""
    n = 100
    ts = np.linspace(0, 1, n)
    accel = np.zeros((n, 3))
    accel[:, 0] = np.sin(np.linspace(0, 2 * np.pi, n))  # varying signal so MAD > 0
    gyro = np.zeros((n, 3))
    # Inject extreme outlier at first sample
    accel[0, 0] = 1e6

    data = {"timestamps": ts, "accel": accel, "gyro": gyro}
    result = cleaner.clean_imu(data)

    # Repaired value should be clamped to the first inlier (index 1), not extrapolated to absurdity
    repaired_val = result["accel"][0, 0]
    assert abs(repaired_val) < 10.0
    # Should equal the nearest inlier value (clamped, not extrapolated)
    assert repaired_val == pytest.approx(accel[1, 0])


def test_clean_imu_backward_timestamp_jump_recorded():
    """Large backward timestamp jump should be recorded in return data."""
    ts = np.array([0.0, 0.01, 0.02, 0.01, 0.03])  # 0.01s backward jump
    accel = np.zeros((5, 3))
    gyro = np.zeros((5, 3))
    data = {"timestamps": ts, "accel": accel, "gyro": gyro}

    result = cleaner.clean_imu(data)
    jumps = result["timestamp_jumps"]
    assert len(jumps) == 1
    assert jumps[0][0] == 3  # index of backward jump
    assert jumps[0][1] == 0.02  # prev_ts
    assert jumps[0][2] == 0.01  # curr_ts
    assert jumps[0][3] == pytest.approx(-0.01)  # delta


def test_clean_imu_tiny_timestamp_jitter_no_record():
    """Floating-point level jitter should not be recorded."""
    ts = np.array([0.0, 0.01, 0.02, 0.0199999, 0.03])  # sub-1e-4 jitter
    accel = np.zeros((5, 3))
    gyro = np.zeros((5, 3))
    data = {"timestamps": ts, "accel": accel, "gyro": gyro}

    result = cleaner.clean_imu(data)
    assert len(result["timestamp_jumps"]) == 0
    # Timestamps should still be strictly monotonic after repair
    assert np.all(np.diff(result["timestamps"]) > 0)
