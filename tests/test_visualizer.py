import matplotlib

matplotlib.use("Agg")

import numpy as np

from imu_camera_sync import cleaner, synchronizer, visualizer

_RNG = np.random.default_rng(42)


def make_imu_data(n=100):
    ts = np.arange(n, dtype=np.float64) / 100.0
    accel = _RNG.standard_normal((n, 3), dtype=np.float64)
    gyro = _RNG.standard_normal((n, 3), dtype=np.float64)
    return {"timestamps": ts, "accel": accel, "gyro": gyro}


def make_cam_data(n=30):
    rng = np.random.default_rng(42)
    # Add small jitter to simulate real frame timestamp variation
    ts = np.arange(n, dtype=np.float64) / 30.0 + rng.normal(0, 1e-6, n)
    return {"timestamps": ts, "fps": 30.0, "frame_count": n}


def test_plot_timestamps():
    imu = make_imu_data()
    cam = make_cam_data()
    fig = visualizer.plot_timestamps(imu, cam, show=False)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_sync_quality():
    imu = make_imu_data(200)
    cam = make_cam_data(60)
    imu_c = cleaner.clean_imu(imu)
    synced = synchronizer.align(imu_c, cam, method="nearest")

    fig = visualizer.plot_sync_quality(imu_c, cam, synced, show=False)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
