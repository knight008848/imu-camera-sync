from pathlib import Path

import numpy as np

from imu_camera_sync import pipeline


def _make_session_dir(tmp_path, with_odometry=True, n_frames=10, name="test_session"):
    """Create a minimal session directory with imu.csv, rgb.mp4, odometry.csv."""
    import cv2

    session = tmp_path / name
    session.mkdir()

    # ── imu.csv (100Hz, slightly longer than camera) ──
    rng = np.random.default_rng(42)
    n_imu = (n_frames * 2) + 10  # ~2x camera, plus padding
    imu_ts = np.arange(n_imu, dtype=np.float64) / 100.0
    accel = rng.standard_normal((n_imu, 3), dtype=np.float64)
    gyro = rng.standard_normal((n_imu, 3), dtype=np.float64)

    imu_lines = ["timestamp,a_x,a_y,a_z,alpha_x,alpha_y,alpha_z"]
    for i in range(n_imu):
        imu_lines.append(
            f"{imu_ts[i]},{accel[i][0]},{accel[i][1]},{accel[i][2]},{gyro[i][0]},{gyro[i][1]},{gyro[i][2]}"
        )
    (session / "imu.csv").write_text("\n".join(imu_lines))

    # ── rgb.mp4 ──
    mp4_path = str(session / "rgb.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, 30.0, (64, 64))
    for _ in range(n_frames):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    if with_odometry:
        cam_ts = np.arange(n_frames, dtype=np.float64) / 30.0
        odom_lines = ["timestamp"]
        for t in cam_ts:
            odom_lines.append(str(t))
        (session / "odometry.csv").write_text("\n".join(odom_lines))

    return str(session)


class TestProcessSession:
    def test_nearest(self, tmp_path):
        session = _make_session_dir(tmp_path)
        result = pipeline.process_session(session, method="nearest")

        assert result["session"] == "test_session"
        assert result["output"] is not None
        assert result["output"].endswith("aligned.csv")
        assert Path(result["output"]).exists()

        assert result["imu"]["samples"] > 0
        assert result["imu"]["duration_s"] > 0
        assert "interval_median_ms" in result["imu"]

        assert result["camera"]["frames"] == 10
        assert result["camera"]["fps"] == 30.0

        align = result["alignment"]
        assert align["method"] == "nearest"
        assert align["error_median_ms"] >= 0
        assert 0 <= align["pct_under_5ms"] <= 100
        assert align["boundary_frames"] >= 0

    def test_interp(self, tmp_path):
        session = _make_session_dir(tmp_path)
        result = pipeline.process_session(session, method="interp")

        assert result["output"] is not None
        assert result["alignment"]["method"] == "interp"
        assert result["alignment"]["error_median_ms"] == 0.0  # interp has None idx

    def test_missing_imu(self, tmp_path):
        """Directories without imu.csv should be skipped by batch, or error in process."""
        session = tmp_path / "bad_session"
        session.mkdir()
        result = pipeline.process_session(str(session))
        assert "error" in result

    def test_with_visualization(self, tmp_path):
        session = _make_session_dir(tmp_path, n_frames=120)
        result = pipeline.process_session(session, method="nearest", with_visualization=True)
        assert Path(result["output"]).parent.joinpath("timestamps.png").exists()
        assert Path(result["output"]).parent.joinpath("sync_quality.png").exists()


class TestBatchProcess:
    def test_batch(self, tmp_path):
        (tmp_path / "not_a_session").mkdir()
        _make_session_dir(tmp_path, name="s1")
        _make_session_dir(tmp_path, name="s2")
        # not_a_session has no imu.csv -> skipped

        results = pipeline.batch_process(str(tmp_path))
        assert len(results) == 2
        assert results[0]["session"] == "s1"
        assert results[1]["session"] == "s2"
        assert all(r["output"] is not None for r in results)

    def test_empty_data_root(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        results = pipeline.batch_process(str(empty))
        assert results == []
