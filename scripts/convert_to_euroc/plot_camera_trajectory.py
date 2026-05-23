#!/usr/bin/env python3
"""
Plot ORB-SLAM3 EuRoC-style trajectory file (CameraTrajectory.txt / f_*.txt).

Each line: timestamp[ns] tx ty tz qx qy qz qw
(ground truth EuRoC uses the same pose layout; timestamps may differ.)

Requires: numpy, matplotlib
  pip install numpy matplotlib

Example:
  python3 scripts/plot_camera_trajectory.py \\
    -i /root/workspace/ORB_SLAM3/CameraTrajectory.txt \\
    -o /root/workspace/ORB_SLAM3/trajectory_plot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_euroc_trajectory(path: Path) -> np.ndarray:
    """Return array shape (N, 8): ts_ns, tx,ty,tz, qx,qy,qz,qw."""
    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                rows.append([float(x) for x in parts[:8]])
            except ValueError:
                continue
    if not rows:
        return np.zeros((0, 8))
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("CameraTrajectory.txt"),
        help="Trajectory text file (default: CameraTrajectory.txt in cwd)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("trajectory_plot.png"),
        help="Output image path (default: trajectory_plot.png)",
    )
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    ap.add_argument(
        "--no-3d",
        action="store_true",
        help="Only draw XY / XZ / YZ subplots (skip 3D panel)",
    )
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        print(f"Error: file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    data = load_euroc_trajectory(inp)
    if data.shape[0] == 0:
        print(f"Error: no valid pose rows in {inp}", file=sys.stderr)
        sys.exit(1)

    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    ts = data[:, 0]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — register 3d projection
    except ImportError as e:
        print(
            "Error: need matplotlib. Install with: pip install matplotlib numpy\n"
            f"({e})",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.no_3d:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        ax3d = None
    else:
        fig = plt.figure(figsize=(14, 10))
        axes = [
            fig.add_subplot(2, 2, 1),
            fig.add_subplot(2, 2, 2),
            fig.add_subplot(2, 2, 3),
        ]
        ax3d = fig.add_subplot(2, 2, 4, projection="3d")

    def style_2d(ax, xdata, ydata, xl, yl, title):
        ax.plot(xdata, ydata, "-", lw=1.2, alpha=0.85)
        ax.scatter(xdata[0], ydata[0], c="green", s=36, zorder=5, label="start")
        ax.scatter(xdata[-1], ydata[-1], c="red", s=36, zorder=5, label="end")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    style_2d(axes[0], x, y, "x [m]", "y [m]", "Top view (XY)")
    style_2d(axes[1], x, z, "x [m]", "z [m]", "Side view (XZ)")
    style_2d(axes[2], y, z, "y [m]", "z [m]", "Side view (YZ)")

    if ax3d is not None:
        ax3d.plot(x, y, z, "-", lw=1.0, alpha=0.85)
        ax3d.scatter(x[0], y[0], z[0], c="g", s=36, label="start")
        ax3d.scatter(x[-1], y[-1], z[-1], c="r", s=36, label="end")
        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")
        ax3d.set_title("3D path (camera frame / ORB world)")
        ax3d.legend(loc="best", fontsize=8)
        # Similar scale on axes when possible
        try:
            max_range = np.array(
                [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]
            ).max() / 2.0
            mid_x = (x.max() + x.min()) * 0.5
            mid_y = (y.max() + y.min()) * 0.5
            mid_z = (z.max() + z.min()) * 0.5
            ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
            ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
            ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
        except Exception:
            pass

    n = len(x)
    dur = (ts[-1] - ts[0]) * 1e-9
    fig.suptitle(
        f"{inp.name}  |  poses={n}  |  duration≈{dur:.2f}s (from timestamps)",
        fontsize=11,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out} ({n} poses)")


if __name__ == "__main__":
    main()
