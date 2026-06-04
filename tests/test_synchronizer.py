import numpy as np
import pytest

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
    assert synced["dropped"] == 0


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
    assert synced["dropped"] == 0


def test_align_unknown_method():
    imu = make_imu_data()
    cam = make_cam_data()
    with pytest.raises(ValueError):
        synchronizer.align(imu, cam, method="invalid")


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
    assert synced["dropped"] == 0


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


def test_align_nearest_tolerance_drops_stale_matches():
    """Frames whose nearest IMU sample exceeds max_tolerance are dropped."""
    imu = make_imu_data(100)  # 0..0.99 s
    # Camera extends 0.5 s beyond IMU range
    ts = np.linspace(0.0, 1.5, 46)  # 0..1.5 s, ~15 frames past IMU end
    cam = {"timestamps": ts, "fps": 30.0, "frame_count": 46}

    synced = synchronizer.align(imu, cam, method="nearest", max_tolerance=0.05)

    n_expected = len(ts)
    n_survived = len(synced["timestamps"])
    assert n_survived < n_expected
    assert synced["dropped"] == n_expected - n_survived
    # All surviving frames must be within tolerance
    matched_ts = imu["timestamps"][synced["imu_indices"]]
    assert np.all(np.abs(matched_ts - synced["timestamps"]) <= 0.05 + 1e-9)
    # Dropped frames should all be beyond IMU coverage
    assert n_survived > 0


def test_align_nearest_custom_tolerance():
    """max_tolerance parameter controls filtering aggressiveness."""
    imu = make_imu_data(100)  # 0..0.99 s
    ts = np.linspace(0.0, 1.5, 46)
    cam = {"timestamps": ts, "fps": 30.0, "frame_count": 46}

    strict = synchronizer.align(imu, cam, method="nearest", max_tolerance=0.01)
    loose = synchronizer.align(imu, cam, method="nearest", max_tolerance=1.0)

    assert strict["dropped"] >= loose["dropped"]


def test_to_csv_nearest(tmp_path):
    imu = make_imu_data(200)
    cam = make_cam_data(10)
    synced = synchronizer.align(imu, cam, method="nearest")
    out = tmp_path / "aligned.csv"
    synchronizer.to_csv(synced, str(out))
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 11  # header + 10 rows
    assert "imu_index" in lines[0]
    assert "accel_x" in lines[0]


def test_to_csv_interp(tmp_path):
    imu = make_imu_data(200)
    cam = make_cam_data(10)
    synced = synchronizer.align(imu, cam, method="interp")
    out = tmp_path / "aligned.csv"
    synchronizer.to_csv(synced, str(out))
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 11
    assert "imu_index" not in lines[0]


def test_to_csv_values(tmp_path):
    imu = {
        "timestamps": np.array([0.0, 0.01, 0.02]),
        "accel": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
        "gyro": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64),
    }
    cam = {"timestamps": np.array([0.0, 0.02])}
    synced = synchronizer.align(imu, cam, method="nearest")
    out = tmp_path / "aligned.csv"
    synchronizer.to_csv(synced, str(out))
    lines = out.read_text().strip().splitlines()
    # Check first data row values
    row0 = lines[1].split(",")
    assert float(row0[0]) == 0.0
    assert float(row0[1]) == 1.0  # accel_x
    assert float(row0[2]) == 2.0  # accel_y
    assert float(row0[3]) == 3.0  # accel_z
