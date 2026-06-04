"""End-to-end pipeline: load, clean, align, export in a single call."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np

from . import cleaner, loader, synchronizer


def process_session(
    session_dir: str,
    method: str = "nearest",
    with_visualization: bool = False,
    max_tolerance_s: float = 0.05,
) -> dict:
    """Run the full pipeline on a single session directory.

    Expects ``session_dir`` to contain ``imu.csv``, ``rgb.mp4``, and
    optionally ``odometry.csv``.

    Parameters
    ----------
    max_tolerance_s : float
        Max allowed time difference in seconds for nearest-neighbor matching.
        Frames exceeding this are dropped. Default 0.05 s (50 ms).

    Returns a quality-metrics dict with keys:
      - session
      - imu: {samples, duration_s, interval_median_ms, interval_std_ms}
      - camera: {frames, duration_s, fps, gap_count, interval_std_ms}
      - alignment: {method, error_mean_ms, error_median_ms, error_p90_ms,
                    error_p99_ms, pct_under_5ms, boundary_frames, imu_coverage_pct,
                    dropped}
      - output: path to generated aligned.csv, or None on failure
    """
    session_dir = str(session_dir)  # coerce Path
    session_name = Path(session_dir).name
    result: dict = {"session": session_name, "output": None}

    try:
        imu_path = f"{session_dir}/imu.csv"
        camera_path = f"{session_dir}/rgb.mp4"
        odometry_path = Path(session_dir) / "odometry.csv"
        odometry_path = str(odometry_path) if odometry_path.exists() else None

        imu_raw = loader.load_imu(imu_path)
        camera_raw = loader.load_camera(camera_path, odometry_path=odometry_path)

        # ── imu stats ──
        imu_intervals_ms = np.diff(imu_raw["timestamps"]) * 1000
        result["imu"] = _stats_imu(imu_raw, imu_intervals_ms)

        # ── camera stats ──
        cam_intervals_ms = np.diff(camera_raw["timestamps"]) * 1000
        expected_interval_ms = 1000.0 / camera_raw["fps"]
        gap_count = int(np.sum(cam_intervals_ms > expected_interval_ms * 1.5))
        result["camera"] = {
            "frames": camera_raw["frame_count"],
            "duration_s": float(camera_raw["timestamps"][-1] - camera_raw["timestamps"][0]),
            "fps": camera_raw["fps"],
            "gap_count": gap_count,
            "interval_std_ms": float(np.std(cam_intervals_ms)),
        }

        # ── clean & align ──
        imu_clean = cleaner.clean_imu(imu_raw)
        camera_clean = cleaner.clean_camera(camera_raw)
        synced = synchronizer.align(
            imu_clean, camera_clean, method=method, max_tolerance_s=max_tolerance_s
        )

        # ── alignment stats ──
        result["alignment"] = _stats_alignment(
            synced, imu_clean, camera_clean, method
        )

        # ── visualization ──
        if with_visualization:
            _save_plots(imu_clean, camera_clean, synced, session_dir)

        # ── export ──
        csv_path = f"{session_dir}/aligned.csv"
        synchronizer.to_csv(synced, csv_path)
        result["output"] = csv_path

    except Exception:
        result["error"] = traceback.format_exc()

    return result


def batch_process(
    data_root: str = "data",
    method: str = "nearest",
    with_visualization: bool = False,
    max_tolerance_s: float = 0.05,
) -> list[dict]:
    """Process every session directory under ``data_root``.

    A session directory must contain ``imu.csv``.  Directories without
    ``imu.csv`` are skipped; individual errors do not stop the batch.
    """
    results: list[dict] = []
    root = Path(data_root)

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "imu.csv").exists():
            continue

        res = process_session(str(entry), method=method,
                              with_visualization=with_visualization,
                              max_tolerance_s=max_tolerance_s)
        results.append(res)

    return results


# ── helpers ──────────────────────────────────────────────────────────────────

def _stats_imu(data: dict, intervals_ms: np.ndarray) -> dict:
    return {
        "samples": len(data["timestamps"]),
        "duration_s": float(data["timestamps"][-1] - data["timestamps"][0]),
        "interval_median_ms": float(np.median(intervals_ms)),
        "interval_std_ms": float(np.std(intervals_ms)),
    }


def _stats_alignment(
    synced: dict, imu: dict, cam: dict, method: str
) -> dict:
    idx = synced.get("imu_indices")
    if idx is not None:
        deltas_ms = np.abs(imu["timestamps"][idx] - synced["timestamps"]) * 1000
        error_mean = float(np.mean(deltas_ms))
        error_median = float(np.median(deltas_ms))
        error_p90 = float(np.percentile(deltas_ms, 90))
        error_p99 = float(np.percentile(deltas_ms, 99))
        pct_under_5ms = float(np.mean(deltas_ms < 5.0) * 100)
        boundary_frames = int(
            np.sum(cam["timestamps"] < imu["timestamps"][0])
            + np.sum(cam["timestamps"] > imu["timestamps"][-1])
        )
        imu_coverage_pct = float(len(np.unique(idx)) / len(imu["timestamps"]) * 100)
    else:
        error_mean = error_median = error_p90 = error_p99 = 0.0
        pct_under_5ms = 100.0
        boundary_frames = 0
        imu_coverage_pct = 100.0

    return {
        "method": method,
        "error_mean_ms": error_mean,
        "error_median_ms": error_median,
        "error_p90_ms": error_p90,
        "error_p99_ms": error_p99,
        "pct_under_5ms": pct_under_5ms,
        "boundary_frames": boundary_frames,
        "imu_coverage_pct": imu_coverage_pct,
        "dropped": synced.get("dropped", 0),
    }


def _save_plots(imu: dict, cam: dict, synced: dict, session_dir: str) -> None:
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    from . import visualizer

    fig1 = visualizer.plot_timestamps(imu, cam, show=False)
    fig1.savefig(f"{session_dir}/timestamps.png", dpi=150, bbox_inches="tight")

    fig2 = visualizer.plot_sync_quality(imu, cam, synced, show=False)
    fig2.savefig(f"{session_dir}/sync_quality.png", dpi=150, bbox_inches="tight")
