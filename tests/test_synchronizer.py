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
