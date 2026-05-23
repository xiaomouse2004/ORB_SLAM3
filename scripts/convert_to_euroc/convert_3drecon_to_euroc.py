#!/usr/bin/env python3
"""
Convert 3drecon VID_* package (SBS stereo MP4 + cam_ts + optional IMU + camchain-imucam.yaml)
into EuRoC ASL-style mav0/ layout for ORB-SLAM3 stereo_euroc / mono_euroc (cam0).

Expected under --input-dir:
  <stem>.mp4              Side-by-side stereo (left | right)
  <stem>_cam_ts.csv       frame_index,timestamp_ns
  depth_imgs/             optional: {frame_index}.png 16-bit depth PNGs → mav0/depth/data/{timestamp_ns}.png
  depth_imgs/depth_meta.json  optional; output as object keyed by timestamp_ns (string)
  <stem>_imu_raw.csv      optional: timestamp_ns,gyro_*,acc_*
  camchain-imucam.yaml    Kalibr-style intrinsics / extrinsics
  imu.yaml                optional: Kalibr IMU noise / rate / time_offset (imu0:)

If the MP4 pixels are already undistorted (e.g. OpenCV undistort / rectification in the
pipeline), pass --undistorted so distortion_coefficients are forced to zero in sensor.yaml
and ORB_SLAM3_stereo_settings.yaml. Intrinsics in camchain must then match the undistorted
image geometry (resolution, fu/fv/cu/cv).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class ImuOrbParams:
    """ORB-SLAM3 IMU.* yaml fields (continuous-time noise densities, see Tracking.cc)."""

    noise_gyro: float
    noise_acc: float
    gyro_walk: float
    acc_walk: float
    frequency: float
    time_offset_s: float = 0.0
    T_i_b: Optional[np.ndarray] = None
    source: str = "default"

    @classmethod
    def euroc_defaults(cls, frequency: float = 200.0) -> "ImuOrbParams":
        return cls(
            noise_gyro=1.7e-4,
            noise_acc=2.0e-3,
            gyro_walk=1.9393e-05,
            acc_walk=3.0e-3,
            frequency=frequency,
            source="EuRoC defaults",
        )


def _which(name: str) -> Optional[str]:
    p = shutil.which(name)
    return p


def _parse_cam_ts(path: Path) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header in {path}")
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        fi_key = fields.get("frame_index") or fields.get("frame")
        ts_key = fields.get("timestamp_ns") or fields.get("timestamp")
        if not fi_key or not ts_key:
            raise ValueError(
                f"{path}: need columns frame_index (or frame) and timestamp_ns (or timestamp), got {reader.fieldnames}"
            )
        for row in reader:
            rows.append((int(row[fi_key]), int(row[ts_key])))
    rows.sort(key=lambda x: x[0])
    return rows


def _check_consecutive_indices(rows: List[Tuple[int, int]]) -> None:
    for i, (idx, _) in enumerate(rows):
        if idx != i:
            raise ValueError(
                f"frame_index must be contiguous starting at 0; expected {i}, got {idx}"
            )


def _estimate_fps_ns(timestamps_ns: List[int]) -> float:
    if len(timestamps_ns) < 2:
        return 5.0
    dts = np.diff(np.array(timestamps_ns, dtype=np.float64)) * 1e-9
    dts = dts[dts > 1e-9]
    if dts.size == 0:
        return 5.0
    return float(1.0 / float(np.median(dts)))


def _ffmpeg_extract_halves(mp4: Path, tmpdir: Path) -> Tuple[Path, Path]:
    ff = _which("ffmpeg")
    if not ff:
        sys.exit(
            "ffmpeg not found in PATH. Install e.g. `apt-get install -y ffmpeg` and retry."
        )
    left_pat = str(tmpdir / "left_%06d.png")
    right_pat = str(tmpdir / "right_%06d.png")
    # SBS: left = left half, right = right half
    cmd_l = [
        ff,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(mp4),
        "-vf",
        "crop=iw/2:ih:0:0",
        "-start_number",
        "0",
        left_pat,
    ]
    cmd_r = [
        ff,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(mp4),
        "-vf",
        "crop=iw/2:ih:iw/2:0",
        "-start_number",
        "0",
        right_pat,
    ]
    subprocess.run(cmd_l, check=True)
    subprocess.run(cmd_r, check=True)
    return Path(left_pat), Path(right_pat)


def _resolve_depth_imgs_dir(input_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        p = explicit.resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"--depth-imgs-dir not found: {p}")
        return p
    cand = input_dir / "depth_imgs"
    return cand if cand.is_dir() else None


def _install_depth_from_imgs(
    depth_imgs_dir: Path,
    rows: List[Tuple[int, int]],
    depth_data: Path,
) -> None:
    """Copy depth_imgs/{frame}.png → depth/data/{timestamp_ns}.png using cam_ts alignment."""
    depth_data.mkdir(parents=True, exist_ok=True)
    for frame_idx, ts in rows:
        src = depth_imgs_dir / f"{frame_idx}.png"
        if not src.is_file():
            sys.exit(f"Missing depth PNG for frame {frame_idx}: {src}")
        shutil.copy2(src, depth_data / f"{ts}.png")


def _write_depth_meta_with_timestamps(
    meta_src: Optional[Path],
    meta_dst: Path,
    rows: List[Tuple[int, int]],
) -> None:
    """Write depth_meta.json as {timestamp_ns: {frame, d_min, d_max, ...}}."""
    ts_by_frame = {fi: ts for fi, ts in rows}
    meta_out: Dict[str, Dict[str, Any]] = {}

    if meta_src is not None and meta_src.is_file():
        with meta_src.open() as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            # Legacy output or hand-edited: keyed by timestamp; normalize via frame index
            entries = []
            for val in raw.values():
                if isinstance(val, dict) and "frame" in val:
                    entries.append(val)
        elif isinstance(raw, list):
            entries = [e for e in raw if isinstance(e, dict)]
        else:
            sys.exit(
                f"depth_meta.json must be a JSON array or object, got {type(raw).__name__}"
            )
        seen: set[int] = set()
        for entry in entries:
            fi = int(entry["frame"])
            if fi not in ts_by_frame:
                sys.exit(f"depth_meta.json frame {fi} has no entry in *_cam_ts.csv")
            ts = ts_by_frame[fi]
            payload = {k: v for k, v in entry.items() if k != "timestamp"}
            payload["frame"] = fi
            meta_out[str(ts)] = payload
            seen.add(fi)
        missing = set(ts_by_frame.keys()) - seen
        if missing:
            sys.exit(
                f"depth_meta.json missing {len(missing)} frame(s) present in cam_ts "
                f"(e.g. {sorted(missing)[:5]})"
            )
    else:
        for fi, ts in rows:
            meta_out[str(ts)] = {"frame": fi}

    meta_dst.parent.mkdir(parents=True, exist_ok=True)
    meta_dst.write_text(json.dumps(meta_out, indent=2) + "\n")


def _glob_seq(pattern: str) -> List[Path]:
    parent = Path(pattern).parent
    stem_glob = Path(pattern).name.replace("%06d", "*")
    files = sorted(parent.glob(stem_glob))
    return files


def _mat44_from_nested_list(m: List[List[float]]) -> np.ndarray:
    a = np.array(m, dtype=np.float64)
    if a.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {a.shape}")
    return a


def _inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _format_matrix_data_4rows(T: np.ndarray, indent: str = "         ") -> str:
    """OpenCV YAML 4x4 row-major data block (4 lines), like Examples/Stereo/EuRoC.yaml."""
    lines: List[str] = []
    for i in range(4):
        row = ", ".join(f"{float(T[i, j]):.12g}" for j in range(4))
        if i == 0:
            lines.append(f"  data: [{row},")
        elif i < 3:
            lines.append(f"{indent}{row},")
        else:
            lines.append(f"{indent}{row}]")
    return "\n".join(lines)


def _format_euroc_T_bs(T_bs: np.ndarray) -> str:
    """EuRoC T_BS block: data as 4 rows of 4 values (row-major), like official ASL sensor.yaml."""
    body = _format_matrix_data_4rows(T_bs)
    return (
        "T_BS:\n"
        "  cols: 4\n"
        "  rows: 4\n"
        f"{body}"
    )


def _format_stereo_T_c1_c2(T_c1_c2: np.ndarray) -> str:
    """ORB-SLAM3 Stereo.T_c1_c2 opencv-matrix block (4 data rows)."""
    body = _format_matrix_data_4rows(T_c1_c2)
    return (
        "Stereo.T_c1_c2: !!opencv-matrix\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "  dt: f\n"
        f"{body}"
    )


def _resolve_imu_yaml(input_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        p = explicit.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"imu.yaml not found: {p}")
        return p
    for name in ("imu.yaml", "imu0.yaml"):
        p = input_dir / name
        if p.is_file():
            return p
    return None


def _parse_imu_yaml(path: Path) -> ImuOrbParams:
    """Parse Kalibr-style imu.yaml (imu0 block) into ORB-SLAM3 IMU parameters."""
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected YAML mapping at root")
    imu0 = raw.get("imu0", raw)
    if not isinstance(imu0, dict):
        raise ValueError(f"{path}: expected imu0 mapping")

    def _f(key: str, alt: Optional[str] = None) -> float:
        for k in (key, alt) if alt else (key,):
            if k and k in imu0:
                return float(imu0[k])
        raise KeyError(f"{path}: missing imu0.{key}")

    T_i_b = None
    if "T_i_b" in imu0:
        T_i_b = _mat44_from_nested_list(imu0["T_i_b"])

    # Kalibr *_noise_density / *_random_walk match ORB-SLAM3 yaml units (see Tracking.cc).
    return ImuOrbParams(
        noise_gyro=_f("gyroscope_noise_density", "gyro_noise_density"),
        noise_acc=_f("accelerometer_noise_density", "acc_noise_density"),
        gyro_walk=_f("gyroscope_random_walk", "gyro_random_walk"),
        acc_walk=_f("accelerometer_random_walk", "acc_random_walk"),
        frequency=float(imu0.get("update_rate", imu0.get("rate_hz", 200.0))),
        time_offset_s=float(imu0.get("time_offset", 0.0)),
        T_i_b=T_i_b,
        source=str(path),
    )


def _compute_T_b_c1(T_cam0_imu: np.ndarray, imu: ImuOrbParams) -> np.ndarray:
    """ORB-SLAM3 IMU.T_b_c1 (Tbc): left camera -> IMU."""
    Tbc = _inv_T(T_cam0_imu)
    if imu.T_i_b is not None and not np.allclose(imu.T_i_b, np.eye(4), atol=1e-6):
        # imu.yaml T_i_b: p_imu = T_i_b @ p_body (Kalibr). Compose when body != IMU sensor.
        Tbc = _inv_T(imu.T_i_b) @ Tbc
    return Tbc.astype(np.float32)


def _cam_imu_timestamp_shift_ns(cam0: Dict[str, Any], imu: ImuOrbParams) -> int:
    """Shift IMU timestamps into camera clock (Kalibr: t_cam = t_imu + timeshift_cam_imu)."""
    shift_s = float(cam0.get("timeshift_cam_imu", 0.0)) + float(imu.time_offset_s)
    return int(round(shift_s * 1e9))


def _format_imu_T_b_c1(T_b_c1: np.ndarray) -> str:
    """ORB-SLAM3 IMU.T_b_c1: left camera (cam0) to IMU body, 4 data rows."""
    body = _format_matrix_data_4rows(T_b_c1)
    return (
        "# Transformation from left camera (cam0) to IMU body (Tbc in ORB-SLAM3)\n"
        "IMU.T_b_c1: !!opencv-matrix\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "  dt: f\n"
        f"{body}"
    )


def _estimate_imu_rate_hz(imu_csv: Path) -> float:
    """Median IMU sample rate from timestamp_ns column."""
    stamps: List[int] = []
    with imu_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = {h.strip().lower(): h for h in (reader.fieldnames or [])}
        ts_k = fields.get("timestamp_ns") or fields.get("timestamp")
        if not ts_k:
            return 200.0
        for row in reader:
            stamps.append(int(float(row[ts_k])))
    if len(stamps) < 2:
        return 200.0
    dts = np.diff(np.array(stamps, dtype=np.float64)) * 1e-9
    dts = dts[dts > 1e-9]
    if dts.size == 0:
        return 200.0
    return float(1.0 / float(np.median(dts)))


def _write_orb_stereo_inertial_yaml(
    out: Path,
    cam0: Dict[str, Any],
    cam1: Dict[str, Any],
    T_c1_c2: np.ndarray,
    T_b_c1: np.ndarray,
    width: int,
    height: int,
    fps: float,
    imu: ImuOrbParams,
    undistorted: bool,
) -> None:
    i0 = cam0["intrinsics"]
    d0 = _distortion_four(cam0, undistorted)
    i1 = cam1["intrinsics"]
    d1 = _distortion_four(cam1, undistorted)
    stereo_T_block = _format_stereo_T_c1_c2(np.asarray(T_c1_c2, dtype=np.float64))
    imu_T_block = _format_imu_T_b_c1(np.asarray(T_b_c1, dtype=np.float64))
    text = f"""%YAML:1.0
# Auto-generated from camchain-imucam.yaml + {imu.source} by convert_3drecon_to_euroc.py
# For stereo_inertial_euroc.
File.version: "1.0"

Camera.type: "PinHole"

Camera1.fx: {_settings_yaml_float(i0[0])}
Camera1.fy: {_settings_yaml_float(i0[1])}
Camera1.cx: {_settings_yaml_float(i0[2])}
Camera1.cy: {_settings_yaml_float(i0[3])}

Camera1.k1: {_settings_yaml_float(d0[0])}
Camera1.k2: {_settings_yaml_float(d0[1])}
Camera1.p1: {_settings_yaml_float(d0[2])}
Camera1.p2: {_settings_yaml_float(d0[3])}

Camera2.fx: {_settings_yaml_float(i1[0])}
Camera2.fy: {_settings_yaml_float(i1[1])}
Camera2.cx: {_settings_yaml_float(i1[2])}
Camera2.cy: {_settings_yaml_float(i1[3])}

Camera2.k1: {_settings_yaml_float(d1[0])}
Camera2.k2: {_settings_yaml_float(d1[1])}
Camera2.p1: {_settings_yaml_float(d1[2])}
Camera2.p2: {_settings_yaml_float(d1[3])}

Camera.width: {int(width)}
Camera.height: {int(height)}
Camera.fps: {int(round(max(1.0, fps)))}
Camera.RGB: 1

Stereo.ThDepth: 60.0
{stereo_T_block}

{imu_T_block}

# IMU noise from {imu.source} (must be YAML reals, not ints, for ORB-SLAM3 Settings)
IMU.NoiseGyro: {_settings_yaml_float(imu.noise_gyro)}
IMU.NoiseAcc: {_settings_yaml_float(imu.noise_acc)}
IMU.GyroWalk: {_settings_yaml_float(imu.gyro_walk)}
IMU.AccWalk: {_settings_yaml_float(imu.acc_walk)}
IMU.Frequency: {_settings_yaml_float(imu.frequency)}

ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""
    out.write_text(text)


def _write_sensor_yaml(
    out_path: Path,
    comment: str,
    T_bs: np.ndarray,
    intrinsics: List[float],
    dist: List[float],
    resolution: Tuple[int, int],
    rate_hz: float,
) -> None:
    fu, fv, cu, cy = intrinsics
    k1, k2, p1, p2 = dist[:4]
    w, h = resolution
    text = f"""# General sensor definitions.
sensor_type: camera
comment: {comment}

# Sensor extrinsics wrt. the body-frame (body = IMU). T_BS = inv(T_cam_imu) from camchain.
{_format_euroc_T_bs(T_bs)}

# Camera specific definitions.
rate_hz: {rate_hz:.6g}
resolution: [{w}, {h}]
camera_model: pinhole
intrinsics: [{fu:.12g}, {fv:.12g}, {cu:.12g}, {cy:.12g}] #fu, fv, cu, cv
distortion_model: radial-tangential
distortion_coefficients: [{k1:.12g}, {k2:.12g}, {p1:.12g}, {p2:.12g}]
"""
    out_path.write_text(text)


def _write_imu_sensor_yaml(out_path: Path, comment: str, imu: ImuOrbParams) -> None:
    """EuRoC ASL mav0/imu0/sensor.yaml (noise densities match Kalibr imu.yaml)."""
    if imu.T_i_b is not None:
        T_bs = _inv_T(imu.T_i_b)
    else:
        T_bs = np.eye(4)
    text = f"""# General sensor definitions.
sensor_type: imu
comment: {comment}

# Sensor extrinsics wrt. the body-frame.
{_format_euroc_T_bs(T_bs)}

# Inertial sensor definitions.
rate_hz: {imu.frequency:.6g}

# Inertial sensor noise model parameters (static)
gyroscope_noise_density: {imu.noise_gyro:.12g}     # [ rad / s / sqrt(Hz) ]
gyroscope_random_walk: {imu.gyro_walk:.12g}       # [ rad / s^2 / sqrt(Hz) ]
accelerometer_noise_density: {imu.noise_acc:.12g}  # [ m / s^2 / sqrt(Hz) ]
accelerometer_random_walk: {imu.acc_walk:.12g}    # [ m / s^3 / sqrt(Hz) ]
"""
    out_path.write_text(text)


def _write_body_yaml(out: Path) -> None:
    out.write_text("comment: Converted from 3drecon (body frame aligned with IMU in camchain)\n")


def _write_data_csv(cam_dir: Path, stamps: List[int]) -> None:
    lines = ["#timestamp [ns],filename"]
    for ts in stamps:
        lines.append(f"{ts},{ts}.png")
    (cam_dir / "data.csv").write_text("\n".join(lines) + "\n")


def _distortion_four(cam: Dict[str, Any], undistorted: bool) -> List[float]:
    if undistorted:
        return [0.0, 0.0, 0.0, 0.0]
    d = [float(x) for x in cam["distortion_coeffs"]]
    if len(d) < 4:
        d = d + [0.0] * (4 - len(d))
    return d[:4]


def _settings_yaml_float(x: float) -> str:
    """Format for ORB-SLAM3 Settings: readParameter<float> needs isReal(); plain `0` is parsed as int."""
    v = float(x)
    s = f"{v:.12g}"
    if s in ("0", "-0"):
        return "0.0"
    return s


def _write_orb_stereo_yaml(
    out: Path,
    cam0: Dict[str, Any],
    cam1: Dict[str, Any],
    T_c1_c2: np.ndarray,
    width: int,
    height: int,
    fps: float,
    undistorted: bool,
) -> None:
    i0 = cam0["intrinsics"]
    d0 = _distortion_four(cam0, undistorted)
    i1 = cam1["intrinsics"]
    d1 = _distortion_four(cam1, undistorted)
    stereo_T_block = _format_stereo_T_c1_c2(np.asarray(T_c1_c2, dtype=np.float64))
    text = f"""%YAML:1.0
# Auto-generated from camchain-imucam.yaml by convert_3drecon_to_euroc.py
File.version: "1.0"

Camera.type: "PinHole"

Camera1.fx: {_settings_yaml_float(i0[0])}
Camera1.fy: {_settings_yaml_float(i0[1])}
Camera1.cx: {_settings_yaml_float(i0[2])}
Camera1.cy: {_settings_yaml_float(i0[3])}

Camera1.k1: {_settings_yaml_float(d0[0])}
Camera1.k2: {_settings_yaml_float(d0[1])}
Camera1.p1: {_settings_yaml_float(d0[2])}
Camera1.p2: {_settings_yaml_float(d0[3])}

Camera2.fx: {_settings_yaml_float(i1[0])}
Camera2.fy: {_settings_yaml_float(i1[1])}
Camera2.cx: {_settings_yaml_float(i1[2])}
Camera2.cy: {_settings_yaml_float(i1[3])}

Camera2.k1: {_settings_yaml_float(d1[0])}
Camera2.k2: {_settings_yaml_float(d1[1])}
Camera2.p1: {_settings_yaml_float(d1[2])}
Camera2.p2: {_settings_yaml_float(d1[3])}

Camera.width: {int(width)}
Camera.height: {int(height)}
Camera.fps: {int(round(max(1.0, fps)))}
Camera.RGB: 1

Stereo.ThDepth: 60.0
{stereo_T_block}

ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""
    out.write_text(text)


def _write_orb_mono_yaml(
    out: Path,
    cam0: Dict[str, Any],
    width: int,
    height: int,
    fps: float,
    undistorted: bool,
) -> None:
    """mono_euroc uses cam0 only; Camera.RGB 0 matches OpenCV BGR loaded images."""
    i0 = cam0["intrinsics"]
    d0 = _distortion_four(cam0, undistorted)
    text = f"""%YAML:1.0
# Auto-generated from camchain-imucam.yaml (cam0) by convert_3drecon_to_euroc.py
File.version: "1.0"

Camera.type: "PinHole"

Camera1.fx: {_settings_yaml_float(i0[0])}
Camera1.fy: {_settings_yaml_float(i0[1])}
Camera1.cx: {_settings_yaml_float(i0[2])}
Camera1.cy: {_settings_yaml_float(i0[3])}

Camera1.k1: {_settings_yaml_float(d0[0])}
Camera1.k2: {_settings_yaml_float(d0[1])}
Camera1.p1: {_settings_yaml_float(d0[2])}
Camera1.p2: {_settings_yaml_float(d0[3])}

Camera.width: {int(width)}
Camera.height: {int(height)}
Camera.fps: {int(round(max(1.0, fps)))}
Camera.RGB: 0

ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""
    out.write_text(text)


def _convert_imu_euroc(
    imu_csv: Path,
    out_csv: Path,
    t0_ns: int,
    t1_ns: int,
    timestamp_shift_ns: int = 0,
) -> int:
    """Write EuRoC imu0/data.csv; keep samples in [t0_ns, t1_ns] after time shift."""
    kept = 0
    header = (
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
        "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
    )
    with imu_csv.open(newline="") as fin, out_csv.open("w", newline="") as fout:
        fout.write(header)
        reader = csv.DictReader(fin)
        fields = {h.strip().lower(): h for h in (reader.fieldnames or [])}
        ts_k = fields.get("timestamp_ns") or fields.get("timestamp")
        if not ts_k:
            raise ValueError(f"{imu_csv}: missing timestamp_ns column")
        gx = fields.get("gyro_x") or fields.get("w_x")
        gy = fields.get("gyro_y") or fields.get("w_y")
        gz = fields.get("gyro_z") or fields.get("w_z")
        ax = fields.get("acc_x") or fields.get("a_x")
        ay = fields.get("acc_y") or fields.get("a_y")
        az = fields.get("acc_z") or fields.get("a_z")
        if not all((gx, gy, gz, ax, ay, az)):
            raise ValueError(f"{imu_csv}: need gyro_* and acc_* columns")
        for row in reader:
            ts = int(float(row[ts_k])) + timestamp_shift_ns
            if ts < t0_ns or ts > t1_ns:
                continue
            fout.write(
                f"{ts},{float(row[gx])},{float(row[gy])},{float(row[gz])},"
                f"{float(row[ax])},{float(row[ay])},{float(row[az])}\n"
            )
            kept += 1
    return kept


def resolve_input_bundle(input_dir: Path) -> Tuple[str, Path, Path]:
    csvs = sorted(input_dir.glob("*_cam_ts.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_cam_ts.csv under {input_dir}")
    cam_ts = csvs[0]
    stem = cam_ts.name.replace("_cam_ts.csv", "")
    mp4 = input_dir / f"{stem}.mp4"
    if not mp4.is_file():
        raise FileNotFoundError(f"Expected video {mp4}")
    return stem, mp4, cam_ts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing <stem>.mp4, <stem>_cam_ts.csv, camchain-imucam.yaml",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Created layout: <output-dir>/mav0/...",
    )
    ap.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete temporary ffmpeg frame dumps (debug).",
    )
    ap.add_argument(
        "--skip-orb-settings",
        action="store_true",
        help="Do not write ORB_SLAM3_stereo_settings.yaml / ORB_SLAM3_mono_settings.yaml next to mav0.",
    )
    ap.add_argument(
        "--imu-margin-ns",
        type=int,
        default=500_000_000,
        help="Include IMU rows within first/last camera timestamp +/- this margin (default 0.5s).",
    )
    ap.add_argument(
        "--undistorted",
        action="store_true",
        help="MP4 frames are already undistorted: set k1,k2,p1,p2 to 0 in EuRoC sensor.yaml and ORB settings (intrinsics must match undistorted images).",
    )
    ap.add_argument(
        "--imu-yaml",
        type=Path,
        default=None,
        help="Kalibr imu.yaml (default: <input-dir>/imu.yaml). Used for ORB IMU noise/rate and timestamp shift.",
    )
    ap.add_argument(
        "--no-depth",
        action="store_true",
        help="Do not copy depth_imgs/ into mav0/depth even if present.",
    )
    ap.add_argument(
        "--depth-imgs-dir",
        type=Path,
        default=None,
        help="Override path to depth PNG folder (default: <input-dir>/depth_imgs).",
    )
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    out_root = args.output_dir.resolve()
    if not input_dir.is_dir():
        sys.exit(f"input-dir is not a directory: {input_dir}")

    stem, mp4, cam_ts_path = resolve_input_bundle(input_dir)
    camchain_path = input_dir / "camchain-imucam.yaml"
    if not camchain_path.is_file():
        sys.exit(f"Missing {camchain_path}")

    rows = _parse_cam_ts(cam_ts_path)
    if not rows:
        sys.exit(f"No rows in {cam_ts_path}")
    _check_consecutive_indices(rows)
    stamps = [ts for _, ts in rows]

    with camchain_path.open() as f:
        camchain = yaml.safe_load(f)
    cam0 = camchain["cam0"]
    cam1 = camchain["cam1"]
    T_cam0_imu = _mat44_from_nested_list(cam0["T_cam_imu"])
    T_cam1_imu = _mat44_from_nested_list(cam1["T_cam_imu"])
    T_c0_c1 = _mat44_from_nested_list(cam1["T_cn_cnm1"])
    imu_in_path = input_dir / f"{stem}_imu_raw.csv"

    w0, h0 = int(cam0["resolution"][0]), int(cam0["resolution"][1])
    w1, h1 = int(cam1["resolution"][0]), int(cam1["resolution"][1])
    if (w0, h0) != (w1, h1):
        print(
            f"Warning: cam0 resolution {w0}x{h0} != cam1 {w1}x{h1}; using cam0 for ORB yaml.",
            file=sys.stderr,
        )

    fps = _estimate_fps_ns(stamps)

    imu_yaml_path = _resolve_imu_yaml(input_dir, args.imu_yaml)
    if imu_yaml_path is not None:
        imu_params = _parse_imu_yaml(imu_yaml_path)
        print(
            f"IMU params from {imu_yaml_path.name}: "
            f"NoiseGyro={imu_params.noise_gyro:g}, NoiseAcc={imu_params.noise_acc:g}, "
            f"GyroWalk={imu_params.gyro_walk:g}, AccWalk={imu_params.acc_walk:g}, "
            f"Frequency={imu_params.frequency:g} Hz, time_offset={imu_params.time_offset_s:g} s"
        )
    else:
        imu_params = ImuOrbParams.euroc_defaults()
        print(
            "No imu.yaml; ORB stereo-inertial yaml uses EuRoC default IMU noise, "
            "Frequency=200 Hz.",
            file=sys.stderr,
        )

    imu_shift_ns = _cam_imu_timestamp_shift_ns(cam0, imu_params)
    if imu_shift_ns != 0:
        print(
            f"IMU timestamp shift: {imu_shift_ns / 1e9:.6f} s "
            f"(camchain timeshift_cam_imu + imu.yaml time_offset)"
        )

    tmpdir = Path(tempfile.mkdtemp(prefix="euroc_conv_"))
    try:
        left_pat, right_pat = _ffmpeg_extract_halves(mp4, tmpdir)
        left_files = _glob_seq(str(left_pat))
        right_files = _glob_seq(str(right_pat))
        if len(left_files) != len(stamps) or len(right_files) != len(stamps):
            sys.exit(
                f"Frame count mismatch: ffmpeg left={len(left_files)} right={len(right_files)} vs cam_ts={len(stamps)}"
            )

        mav0 = out_root / "mav0"
        cam0_data = mav0 / "cam0" / "data"
        cam1_data = mav0 / "cam1" / "data"
        imu0_dir = mav0 / "imu0"
        for d in (cam0_data, cam1_data, imu0_dir):
            d.mkdir(parents=True, exist_ok=True)

        for i, ts in enumerate(stamps):
            name = f"{ts}.png"
            shutil.copy2(left_files[i], cam0_data / name)
            shutil.copy2(right_files[i], cam1_data / name)

        _write_data_csv(mav0 / "cam0", stamps)
        _write_data_csv(mav0 / "cam1", stamps)

        depth_imgs_dir = None if args.no_depth else _resolve_depth_imgs_dir(
            input_dir, args.depth_imgs_dir
        )
        if depth_imgs_dir is not None:
            depth_data = mav0 / "depth" / "data"
            _install_depth_from_imgs(depth_imgs_dir, rows, depth_data)
            _write_data_csv(mav0 / "depth", stamps)
            meta_src = depth_imgs_dir / "depth_meta.json"
            meta_dst = mav0 / "depth" / "depth_meta.json"
            _write_depth_meta_with_timestamps(
                meta_src if meta_src.is_file() else None, meta_dst, rows
            )
            print(
                f"Wrote depth images to {depth_data} ({len(rows)} frames, from {depth_imgs_dir})"
            )
            print(f"Wrote {meta_dst} (dict keyed by timestamp_ns)")
        elif not args.no_depth:
            print(
                f"No depth_imgs/ under {input_dir}; skipped mav0/depth.",
                file=sys.stderr,
            )

        T_bs0 = _inv_T(T_cam0_imu)
        T_bs1 = _inv_T(T_cam1_imu)
        d0 = _distortion_four(cam0, args.undistorted)
        d1 = _distortion_four(cam1, args.undistorted)
        cam0_yaml_comment = f"converted cam0 ({stem})"
        cam1_yaml_comment = f"converted cam1 ({stem})"
        if args.undistorted:
            cam0_yaml_comment += ", undistorted pixels (distortion forced 0)"
            cam1_yaml_comment += ", undistorted pixels (distortion forced 0)"
        _write_sensor_yaml(
            mav0 / "cam0" / "sensor.yaml",
            cam0_yaml_comment,
            T_bs0,
            [float(x) for x in cam0["intrinsics"]],
            d0,
            (w0, h0),
            fps,
        )
        _write_sensor_yaml(
            mav0 / "cam1" / "sensor.yaml",
            cam1_yaml_comment,
            T_bs1,
            [float(x) for x in cam1["intrinsics"]],
            d1,
            (w1, h1),
            fps,
        )
        _write_body_yaml(mav0 / "body.yaml")

        imu_sensor_comment = f"converted imu0 ({stem})"
        if imu_yaml_path is not None:
            imu_sensor_comment += f", from {imu_yaml_path.name}"
        else:
            imu_sensor_comment += ", EuRoC default noise (no imu.yaml)"
        _write_imu_sensor_yaml(mav0 / "imu0" / "sensor.yaml", imu_sensor_comment, imu_params)
        print(f"Wrote {mav0 / 'imu0' / 'sensor.yaml'}")

        # ORB stereo: c1 = left (cam0), c2 = right (cam1).
        # Kalibr cam1 T_cn_cnm1 is T_cam1_cam0; ORB-SLAM3 Stereo.T_c1_c2 expects T_cam0_cam1.
        T_c1_c2 = _inv_T(T_c0_c1).astype(np.float32)

        ts_txt = out_root / "stereo_timestamps_ns.txt"
        ts_txt.write_text("\n".join(str(ts) for ts in stamps) + "\n")

        if imu_in_path.is_file():
            t0 = stamps[0] - args.imu_margin_ns + imu_shift_ns
            t1 = stamps[-1] + args.imu_margin_ns + imu_shift_ns
            imu_out = imu0_dir / "data.csv"
            n = _convert_imu_euroc(
                imu_in_path, imu_out, t0, t1, timestamp_shift_ns=imu_shift_ns
            )
            print(f"Wrote imu0/data.csv with {n} samples (time window around cameras).")
            if n > 1:
                measured_hz = _estimate_imu_rate_hz(imu_out)
                if abs(measured_hz - imu_params.frequency) / imu_params.frequency > 0.15:
                    print(
                        f"Warning: imu.yaml update_rate={imu_params.frequency:.2f} Hz but "
                        f"measured {measured_hz:.2f} Hz in output; keeping yaml value for "
                        "IMU.Frequency.",
                        file=sys.stderr,
                    )
        else:
            print(f"No {imu_in_path.name}; skipped imu0/data.csv", file=sys.stderr)

        T_b_c1 = _compute_T_b_c1(T_cam0_imu, imu_params)

        if not args.skip_orb_settings:
            out_stereo = out_root / "ORB_SLAM3_stereo_settings.yaml"
            _write_orb_stereo_yaml(
                out_stereo, cam0, cam1, T_c1_c2, w0, h0, fps, args.undistorted
            )
            print(f"Wrote {out_stereo}")
            out_mono = out_root / "ORB_SLAM3_mono_settings.yaml"
            _write_orb_mono_yaml(out_mono, cam0, w0, h0, fps, args.undistorted)
            print(f"Wrote {out_mono}")
            if imu_in_path.is_file():
                out_si = out_root / "ORB_SLAM3_stereo_inertial_settings.yaml"
                _write_orb_stereo_inertial_yaml(
                    out_si,
                    cam0,
                    cam1,
                    T_c1_c2,
                    T_b_c1,
                    w0,
                    h0,
                    fps,
                    imu_params,
                    args.undistorted,
                )
                print(
                    f"Wrote {out_si} (stereo_inertial_euroc, IMU.Frequency={imu_params.frequency:.2f} Hz)"
                )

        print(f"Done. EuRoC tree at {mav0}")
        print(f"ORB-SLAM3 stereo timestamps file: {ts_txt}")
        if args.undistorted:
            print(
                "Note: --undistorted was set: k1..p2 are 0. "
                "camchain intrinsics and resolution must match the undistorted MP4 pixels."
            )
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
