#!/usr/bin/env python3
"""
Convert EuRoC-style dataset + ORB-SLAM3 trajectory to COLMAP format for FastGS.

Input (EuRoC layout):
  <euroc_root>/mav0/cam0/data/*.png
  <euroc_root>/mav0/cam0/data.csv
  <euroc_root>/mav0/cam0/sensor.yaml
  <euroc_root>/CameraTrajectory.txt       # ORB-SLAM3 output (required)
  <euroc_root>/mav0/depth/data/*.png      # 16-bit depth (optional)
  <euroc_root>/mav0/depth/depth_meta.json
  See <euroc_root>/depth_readme.md for depth decoding.

Output (COLMAP / FastGS layout):
  <output_dir>/images/
  <output_dir>/sparse/0/cameras.txt
  <output_dir>/sparse/0/images.txt
  <output_dir>/sparse/0/points3D.txt
  <output_dir>/sparse/0/points3D.ply
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required: pip install pillow") from exc


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (COLMAP convention: qvec = [qw, qx, qy, qz])
# ---------------------------------------------------------------------------

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def rotmat2qvec(rot: np.ndarray) -> np.ndarray:
    rot = rot.astype(np.float64)
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rot.flat
    k = np.array([
        [rxx - ryy - rzz, 0.0, 0.0, 0.0],
        [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def quat_xyzw_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return qvec2rotmat(np.array([qw, qx, qy, qz], dtype=np.float64))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(dot)
    s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return s0 * q0 + s1 * q1


def tum_to_colmap(tx: float, ty: float, tz: float,
                  qx: float, qy: float, qz: float, qw: float) -> tuple[np.ndarray, np.ndarray]:
    """ORB-SLAM3 TUM trajectory stores T_wc (camera pose in world)."""
    r_wc = quat_xyzw_to_rotmat(qx, qy, qz, qw)
    t_wc = np.array([tx, ty, tz], dtype=np.float64)
    r_cw = r_wc.T
    t_cw = -r_wc.T @ t_wc
    qvec = rotmat2qvec(r_cw)
    return qvec, t_cw


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_sensor_yaml(sensor_yaml: Path) -> dict:
    with open(sensor_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    intr = data["intrinsics"]
    width, height = data["resolution"]
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(intr[0]),
        "fy": float(intr[1]),
        "cx": float(intr[2]),
        "cy": float(intr[3]),
        "T_BS": np.array(data["T_BS"]["data"], dtype=np.float64).reshape(4, 4),
    }


def load_cam0_index(data_csv: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with open(data_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            ts = int(row[0].strip())
            fname = row[1].strip()
            rows.append((ts, fname))
    rows.sort(key=lambda x: x[0])
    return rows


def parse_trajectory(path: Path, fmt: str) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """
    Returns list of (timestamp, qvec_colmap, tvec_colmap).
    Timestamp unit is kept as in file (ns or seconds); caller normalizes.
    """
    entries: list[tuple[float, np.ndarray, np.ndarray]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line.strip())
            parts = [p for p in parts if p]
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            if fmt == "euroc":
                tx, ty, tz = map(float, parts[1:4])
                qw, qx, qy, qz = map(float, parts[4:8])
            else:
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
            qvec, tvec = tum_to_colmap(tx, ty, tz, qx, qy, qz, qw)
            entries.append((ts, qvec, tvec))
    if not entries:
        raise RuntimeError(f"No valid poses found in trajectory: {path}")
    entries.sort(key=lambda x: x[0])
    return entries


def detect_trajectory_format(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "q_RS_w" in line or "p_RS_R" in line:
                return "euroc"
            if line.strip() and not line.startswith("#"):
                break
    return "tum"


def find_trajectory(euroc_root: Path) -> Path | None:
    """ORB-SLAM3 trajectory: default <euroc_root>/CameraTrajectory.txt."""
    for name in ("CameraTrajectory.txt", "KeyFrameTrajectory.txt"):
        p = euroc_root / name
        if p.is_file():
            return p
    return None


def normalize_timestamps_to_ns(traj_ts: np.ndarray, image_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    traj_ns = traj_ts.copy()
    image_ns = image_ts.astype(np.float64)
    if np.max(traj_ns) < 1e12 and np.max(image_ns) > 1e12:
        traj_ns *= 1e9
    elif np.max(traj_ns) > 1e12 and np.max(image_ns) < 1e12:
        image_ns *= 1e9
    return traj_ns, image_ns


def interpolate_poses(
    query_ts: np.ndarray,
    key_ts: np.ndarray,
    key_qvecs: np.ndarray,
    key_tvecs: np.ndarray,
    max_delta_ns: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_out = np.zeros((len(query_ts), 4), dtype=np.float64)
    t_out = np.zeros((len(query_ts), 3), dtype=np.float64)
    valid = np.zeros(len(query_ts), dtype=bool)

    for i, ts in enumerate(query_ts):
        idx = np.searchsorted(key_ts, ts)
        if idx == 0:
            if abs(key_ts[0] - ts) <= max_delta_ns:
                q_out[i] = key_qvecs[0]
                t_out[i] = key_tvecs[0]
                valid[i] = True
            continue
        if idx >= len(key_ts):
            if abs(key_ts[-1] - ts) <= max_delta_ns:
                q_out[i] = key_qvecs[-1]
                t_out[i] = key_tvecs[-1]
                valid[i] = True
            continue

        t0, t1 = key_ts[idx - 1], key_ts[idx]
        if ts < t0 or ts > t1:
            continue
        if (ts - t0) > max_delta_ns or (t1 - ts) > max_delta_ns:
            nearest = idx - 1 if (ts - t0) < (t1 - ts) else idx
            if abs(key_ts[nearest] - ts) <= max_delta_ns:
                q_out[i] = key_qvecs[nearest]
                t_out[i] = key_tvecs[nearest]
                valid[i] = True
            continue

        alpha = 0.0 if t1 == t0 else (ts - t0) / (t1 - t0)
        q_out[i] = slerp(key_qvecs[idx - 1], key_qvecs[idx], alpha)
        t_out[i] = (1.0 - alpha) * key_tvecs[idx - 1] + alpha * key_tvecs[idx]
        valid[i] = True

    return q_out, t_out, valid


def load_depth_meta(meta_path: Path) -> dict[str, dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if isinstance(meta, list):
        return {str(entry["frame"]): entry for entry in meta}
    return {str(k): v for k, v in meta.items()}


def decode_depth_uint16(pixels: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """16-bit PNG -> depth in meters (see depth_readme.md)."""
    pv = pixels.astype(np.float32)
    depth = np.zeros_like(pv)
    valid = pv > 0
    if not np.any(valid):
        return depth
    normalized = (pv[valid] - 1.0) / 65534.0
    depth[valid] = d_min + (1.0 - normalized) * (d_max - d_min)
    return depth


def colmap_pose_to_world(
    qvec: np.ndarray, tvec: np.ndarray, points_cam: np.ndarray
) -> np.ndarray:
    """Unproject camera-frame points to world (COLMAP world-to-camera convention)."""
    r_cw = qvec2rotmat(qvec)
    r_wc = r_cw.T
    cam_center = -r_wc @ tvec
    return (r_wc @ points_cam.T).T + cam_center


def generate_initial_pointcloud_from_depth(
    filtered_index: list[tuple[int, str]],
    qvecs: np.ndarray,
    tvecs: np.ndarray,
    image_dir: Path,
    depth_dir: Path,
    depth_meta: dict[str, dict],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_min: float,
    depth_max: float,
    num_points: int,
    seed: int,
    pixel_stride: int,
    frame_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depth maps (0.3–10 m) with RGB colors for FastGS initialization."""
    rng = np.random.default_rng(seed)
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    frame_indices = range(0, len(filtered_index), max(1, frame_step))
    for fi in frame_indices:
        ts, fname = filtered_index[fi]
        stem = Path(fname).stem
        if stem not in depth_meta:
            raise KeyError(
                f"Depth meta missing for {stem}. "
                f"Check {depth_dir.parent / 'depth_meta.json'} keys match cam0 filenames."
            )

        depth_path = depth_dir / fname
        rgb_path = image_dir / fname
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing depth image: {depth_path}")
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB image: {rgb_path}")

        d_min = float(depth_meta[stem]["d_min"])
        d_max = float(depth_meta[stem]["d_max"])

        depth_png = np.array(Image.open(depth_path))
        depth_m = decode_depth_uint16(depth_png, d_min, d_max)

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        h, w = depth_m.shape
        stride = max(1, pixel_stride)
        vs = np.arange(0, h, stride)
        us = np.arange(0, w, stride)
        uu, vv = np.meshgrid(us, vs)

        z = depth_m[vv, uu]
        mask = (z >= depth_min) & (z <= depth_max)
        if not np.any(mask):
            continue

        u_f = uu[mask].astype(np.float64)
        v_f = vv[mask].astype(np.float64)
        z_f = z[mask]

        x_cam = (u_f - cx) * z_f / fx
        y_cam = (v_f - cy) * z_f / fy
        points_cam = np.stack([x_cam, y_cam, z_f], axis=1)

        qvec = qvecs[fi]
        tvec = tvecs[fi]
        points_world = colmap_pose_to_world(qvec, tvec, points_cam)

        colors = rgb[vv, uu][mask]
        print("The colors size is " + str(len(colors)))
        all_points.append(points_world)
        all_colors.append(colors)


    if not all_points:
        raise RuntimeError(
            f"No valid depth points in [{depth_min}, {depth_max}] m. "
            "Check depth maps, meta, or pose alignment."
        )

    points = np.vstack(all_points)
    colors = np.vstack(all_colors).astype(np.uint8)
    print("The all_points size is " + str(len(points)))

    if len(points) > num_points:
        idx = rng.choice(len(points), num_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors


def generate_initial_pointcloud_synthetic(
    qvecs: np.ndarray,
    tvecs: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    num_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: synthetic points along camera rays (gray color)."""
    rng = np.random.default_rng(seed)
    cam_centers = []
    for qvec, tvec in zip(qvecs, tvecs):
        r = qvec2rotmat(qvec).T
        c2w = np.eye(4)
        c2w[:3, :3] = r
        c2w[:3, 3] = -r @ tvec
        cam_centers.append(c2w[:3, 3])
    cam_centers = np.stack(cam_centers, axis=0)
    center = cam_centers.mean(axis=0)
    radius = np.linalg.norm(cam_centers - center, axis=1).max()
    radius = max(radius, 0.5)

    points = []
    step = max(1, len(qvecs) // max(1, num_points // 100))
    for qvec, tvec in zip(qvecs[::step], tvecs[::step]):
        r_wc = qvec2rotmat(qvec).T
        cam_center = -r_wc @ tvec
        forward = r_wc[:, 2]
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        for depth in np.linspace(0.3 * radius, 2.0 * radius, 20):
            points.append(cam_center + forward * depth)
    points = np.array(points, dtype=np.float64)

    remaining = max(0, num_points - len(points))
    if remaining > 0:
        rand = center + rng.uniform(-radius, radius, size=(remaining, 3))
        points = np.vstack([points, rand]) if len(points) else rand

    if len(points) > num_points:
        idx = rng.choice(len(points), num_points, replace=False)
        points = points[idx]

    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    return points, colors


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} 0 0 0 {int(r)} {int(g)} {int(b)}\n")


def write_colmap_text(
    sparse_dir: Path,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_names: list[str],
    qvecs: np.ndarray,
    tvecs: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)

    with open(sparse_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    with open(sparse_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_names)}\n")
        for i, name in enumerate(image_names, start=1):
            q = qvecs[i - 1]
            t = tvecs[i - 1]
            f.write(
                f"{i} {q[0]:.16f} {q[1]:.16f} {q[2]:.16f} {q[3]:.16f} "
                f"{t[0]:.16f} {t[1]:.16f} {t[2]:.16f} 1 {name}\n\n"
            )

    with open(sparse_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for i, ((x, y, z), (r, g, b)) in enumerate(zip(points, colors), start=1):
            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} 0.0\n")

    write_ply(sparse_dir / "points3D.ply", points, colors)


def link_or_copy_images(
    src_dir: Path,
    dst_dir: Path,
    entries: list[tuple[int, str]],
    copy_images: bool,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for _, fname in entries:
        src = src_dir / fname
        dst = dst_dir / fname
        if not src.is_file():
            raise FileNotFoundError(f"Missing image: {src}")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy_images:
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)


def convert(args: argparse.Namespace) -> None:
    euroc_root = Path(args.euroc_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    cam0_dir = euroc_root / "mav0" / "cam0"
    data_csv = cam0_dir / "data.csv"
    sensor_yaml = cam0_dir / "sensor.yaml"
    image_dir = cam0_dir / "data"

    if not data_csv.is_file():
        raise FileNotFoundError(f"Missing cam0 index: {data_csv}")
    if not sensor_yaml.is_file():
        raise FileNotFoundError(f"Missing intrinsics: {sensor_yaml}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing cam0 images: {image_dir}")

    traj_path = find_trajectory(euroc_root)
    if traj_path is None:
        raise FileNotFoundError(
            f"ORB-SLAM3 trajectory not found. Place CameraTrajectory.txt in:\n"
            f"  {euroc_root / 'CameraTrajectory.txt'}"
        )

    traj_fmt = args.trajectory_format
    if traj_fmt == "auto":
        traj_fmt = detect_trajectory_format(traj_path)

    sensor = load_sensor_yaml(sensor_yaml)
    image_index = load_cam0_index(data_csv)
    if args.max_images > 0:
        image_index = image_index[: args.max_images]

    traj_entries = parse_trajectory(traj_path, traj_fmt)
    traj_ts = np.array([e[0] for e in traj_entries], dtype=np.float64)
    traj_q = np.stack([e[1] for e in traj_entries], axis=0)
    traj_t = np.stack([e[2] for e in traj_entries], axis=0)

    image_ts = np.array([ts for ts, _ in image_index], dtype=np.float64)
    traj_ts, image_ts = normalize_timestamps_to_ns(traj_ts, image_ts)

    qvecs, tvecs, valid = interpolate_poses(
        image_ts, traj_ts, traj_q, traj_t, max_delta_ns=args.max_time_delta_ns
    )
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        raise RuntimeError(
            "No cam0 frames matched trajectory timestamps. "
            "Check timestamp units or increase --max-time-delta-ns."
        )

    if args.min_valid_ratio > 0:
        ratio = len(valid_indices) / len(image_index)
        if ratio < args.min_valid_ratio:
            raise RuntimeError(
                f"Only {ratio:.1%} frames matched poses (< {args.min_valid_ratio:.1%}). "
                "Use CameraTrajectory.txt (dense) instead of KeyFrameTrajectory.txt, "
                "or increase --max-time-delta-ns."
            )

    filtered_index = [image_index[i] for i in valid_indices]
    filtered_q = qvecs[valid_indices]
    filtered_t = tvecs[valid_indices]
    image_names = [fname for _, fname in filtered_index]

    images_out = output_dir / "images"
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    link_or_copy_images(image_dir, images_out, filtered_index, copy_images=args.copy_images)

    depth_dir = Path(args.depth_dir).resolve() if args.depth_dir else euroc_root / "mav0" / "depth" / "data"
    depth_meta_path = (
        Path(args.depth_meta).resolve()
        if args.depth_meta
        else euroc_root / "mav0" / "depth" / "depth_meta.json"
    )
    use_depth = args.use_depth
    if use_depth and (not depth_dir.is_dir() or not depth_meta_path.is_file()):
        print(
            f"Warning: depth not found ({depth_dir} / {depth_meta_path}), "
            "falling back to synthetic point cloud.",
            file=sys.stderr,
        )
        use_depth = False

    if use_depth:
        depth_meta = load_depth_meta(depth_meta_path)
        points, colors = generate_initial_pointcloud_from_depth(
            filtered_index,
            filtered_q,
            filtered_t,
            image_dir,
            depth_dir,
            depth_meta,
            sensor["fx"],
            sensor["fy"],
            sensor["cx"],
            sensor["cy"],
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            num_points=args.num_init_points,
            seed=args.seed,
            pixel_stride=args.depth_pixel_stride,
            frame_step=args.depth_frame_step,
        )
        init_mode = f"depth back-projection [{args.depth_min}, {args.depth_max}] m"
    else:
        points, colors = generate_initial_pointcloud_synthetic(
            filtered_q,
            filtered_t,
            sensor["fx"],
            sensor["fy"],
            sensor["cx"],
            sensor["cy"],
            num_points=args.num_init_points,
            seed=args.seed,
        )
        init_mode = "synthetic (no depth)"

    sparse_dir = output_dir / "sparse" / "0"
    write_colmap_text(
        sparse_dir,
        sensor["width"],
        sensor["height"],
        sensor["fx"],
        sensor["fy"],
        sensor["cx"],
        sensor["cy"],
        image_names,
        filtered_q,
        filtered_t,
        points,
        colors,
    )

    print(f"EuRoC root:      {euroc_root}")
    print(f"Trajectory:      {traj_path} ({traj_fmt})")
    print(f"Output:          {output_dir}")
    print(f"Images linked:   {len(image_names)} / {len(image_index)} cam0 frames")
    print(f"Init points:     {len(points)} ({init_mode})")
    print(f"Camera:          PINHOLE {sensor['width']}x{sensor['height']}, "
          f"fx={sensor['fx']:.2f}, fy={sensor['fy']:.2f}")
    print("\nFastGS train example:")
    print(f"  python train.py -s {output_dir} -i images --test_iterations 30000")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert EuRoC cam0 + ORB-SLAM3 trajectory to COLMAP for FastGS."
    )
    p.add_argument(
        "--euroc-root",
        required=True,
        help="EuRoC dataset root (mav0/cam0/ + CameraTrajectory.txt).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output COLMAP scene directory for FastGS.",
    )
    p.add_argument(
        "--trajectory-format",
        choices=["auto", "tum", "euroc"],
        default="auto",
        help="Trajectory format. TUM: ts tx ty tz qx qy qz qw. "
             "EuRoC: ts tx ty tz qw qx qy qz.",
    )
    p.add_argument(
        "--max-time-delta-ns",
        type=float,
        default=5e7,
        help="Max allowed timestamp gap (nanoseconds) for pose interpolation / nearest match.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Use only the first N cam0 frames (-1 = all).",
    )
    p.add_argument(
        "--num-init-points",
        type=int,
        default=1000000,
        help="Target number of initial 3D points after subsampling.",
    )
    p.add_argument(
        "--use-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mav0/depth 16-bit PNG + depth_meta.json for colored init point cloud.",
    )
    p.add_argument(
        "--depth-dir",
        default=None,
        help="Depth PNG directory (default: <euroc-root>/mav0/depth/data).",
    )
    p.add_argument(
        "--depth-meta",
        default=None,
        help="depth_meta.json path (default: <euroc-root>/mav0/depth/depth_meta.json).",
    )
    p.add_argument(
        "--depth-min",
        type=float,
        default=0.3,
        help="Minimum valid depth in meters (inclusive).",
    )
    p.add_argument(
        "--depth-max",
        type=float,
        default=10.0,
        help="Maximum valid depth in meters (inclusive).",
    )
    p.add_argument(
        "--depth-pixel-stride",
        type=int,
        default=8,
        help="Subsample every N pixels when back-projecting depth.",
    )
    p.add_argument(
        "--depth-frame-step",
        type=int,
        default=1,
        help="Use every N-th frame for depth fusion (1 = all frames).",
    )
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove output-dir before conversion.",
    )
    p.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.5,
        help="Minimum fraction of cam0 frames that must receive a pose.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for initial point cloud.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    try:
        convert(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
