import numpy as np

from imu_camera_sync import synchronizer


def make_imu_data(n=100):
    ts = np.arange(n, dtype=np.float64) / 100.0
    accel = np.random.randn(n, 3).astype(np.float64)
    gyro = np.random.randn(n, 3).astype(np.float64)
    return {"timestamps": ts, "accel": accel, "gyro": gyro}


def make_cam_data(n=30):
    ts = np.arange(n, dtype=np.float64) / 30.0
    return {"timestamps": ts, "fps": 30.0, "frame_count": n}


def test_align_nearest():
    imu = make_imu_data(200)
    cam = make_cam_data(60)
    synced = synchronizer.align(imu, cam, method="nearest")

    assert synced["accel_aligned"].shape == (60, 3)
    assert synced["gyro_aligned"].shape == (60, 3)
    assert synced["imu_indices"].shape == (60,)
    assert synced["imu_indices"].min() >= 0
    assert synced["imu_indices"].max() < 200


def test_align_nearest_max_error():
    imu = make_imu_data(200)
    cam = make_cam_data(60)

    synced = synchronizer.align(imu, cam, method="nearest")
    matched_imu_ts = imu["timestamps"][synced["imu_indices"]]
    deltas = np.abs(matched_imu_ts - cam["timestamps"])

    # Max error should not exceed half the largest IMU interval
    max_imu_interval = np.max(np.diff(imu["timestamps"])) / 2
    assert np.all(deltas <= max_imu_interval + 1e-9)


def test_align_interp():
    imu = make_imu_data(200)
    cam = make_cam_data(60)
    synced = synchronizer.align(imu, cam, method="interp")

    assert synced["accel_aligned"].shape == (60, 3)
    assert synced["gyro_aligned"].shape == (60, 3)
    assert synced["imu_indices"] is None


def test_align_unknown_method():
    imu = make_imu_data()
    cam = make_cam_data()
    try:
        synchronizer.align(imu, cam, method="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_align_nearest_with_gappy_camera():
    """Nearest-neighbor should handle large gaps in camera timestamps."""
    imu = make_imu_data(200)  # 100 Hz, 0..1.99 s
    ts = np.arange(30, dtype=np.float64) / 30.0  # uniform 1 s
    ts[15:] += 0.5  # introduce a 0.5 s gap at frame 15
    cam = {"timestamps": ts, "fps": 30.0, "frame_count": 30}

    synced = synchronizer.align(imu, cam, method="nearest")
    matched_ts = imu["timestamps"][synced["imu_indices"]]

    assert synced["accel_aligned"].shape == (30, 3)
    assert synced["imu_indices"].min() >= 0
    assert synced["imu_indices"].max() < 200
    # Max error should be bounded by the largest IMU interval
    max_imu_interval = np.max(np.diff(imu["timestamps"]))
    assert np.all(np.abs(matched_ts - cam["timestamps"]) <= max_imu_interval + 1e-9)


def test_align_interp_out_of_range_clamped():
    """Frames outside IMU range should be clamped to boundary values, not extrapolated."""
    imu = make_imu_data(100)  # 0..0.99 s
    # Camera frames extend beyond IMU range
    ts = np.linspace(-0.1, 1.1, 36)  # -0.1..1.1 s, 10 frames outside
    cam = {"timestamps": ts, "fps": 30.0, "frame_count": 36}

    synced = synchronizer.align(imu, cam, method="interp")

    # Frames before IMU range: clamped to imu accel[0]
    out_left = ts < imu["timestamps"][0]  # first 3 frames
    for i in np.where(out_left)[0]:
        np.testing.assert_array_equal(synced["accel_aligned"][i], imu["accel"][0])

    # Frames after IMU range: clamped to imu accel[-1]
    out_right = ts > imu["timestamps"][-1]  # last 3 frames
    for i in np.where(out_right)[0]:
        np.testing.assert_array_equal(synced["accel_aligned"][i], imu["accel"][-1])
