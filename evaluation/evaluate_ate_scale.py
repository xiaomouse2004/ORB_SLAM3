#!/usr/bin/env python3
# Modified by Raul Mur-Artal
# Automatically compute the optimal scale factor for monocular VO/SLAM.
# Ported to Python 3.

"""
Compute absolute trajectory error (ATE) from ground truth and estimated trajectories.
"""

import sys
import numpy as np
import argparse
import associate


def _as_array(x):
    """Convert matrix or ndarray to 1D numpy array."""
    return np.asarray(x).ravel()


def align(model, data):
    """Align two trajectories using Horn's method (closed-form)."""
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh

    rotmodel = rot * model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(
            data_zerocentered[:, column].transpose(), rotmodel[:, column]
        )
        normi = np.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    s = float(dots / norms)

    transGT = data.mean(1) - s * rot * model.mean(1)
    trans = data.mean(1) - rot * model.mean(1)

    model_alignedGT = s * rot * model + transGT
    model_aligned = rot * model + trans

    alignment_errorGT = model_alignedGT - data
    alignment_error = model_aligned - data

    trans_errorGT = np.sqrt(
        np.sum(np.multiply(alignment_errorGT, alignment_errorGT), 0)
    )
    trans_error = np.sqrt(
        np.sum(np.multiply(alignment_error, alignment_error), 0)
    )

    return rot, transGT, _as_array(trans_errorGT), trans, _as_array(trans_error), s


def plot_traj(ax, stamps, traj, style, color, label):
    """Plot a trajectory using matplotlib."""
    stamps = list(stamps)
    stamps.sort()
    interval = np.median([s - t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i] - last < 2 * interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


def _to_xyz_array(traj_dict, stamps, scale=1.0):
    return np.matrix(
        [
            [float(value) * float(scale) for value in traj_dict[b][0:3]]
            for b in stamps
        ]
    ).transpose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ATE from ground truth and estimated trajectories."
    )
    parser.add_argument(
        "first_file",
        help="ground truth (format: timestamp tx ty tz qx qy qz qw)",
    )
    parser.add_argument(
        "second_file",
        help="estimated trajectory (format: timestamp tx ty tz qx qy qz qw)",
    )
    parser.add_argument(
        "--offset",
        help="time offset added to timestamps of second file (default: 0.0)",
        default=0.0,
    )
    parser.add_argument(
        "--scale",
        help="scaling factor for second trajectory (default: 1.0)",
        default=1.0,
    )
    parser.add_argument(
        "--max_difference",
        help="max allowed time difference for matching in ns (default: 20000000)",
        default=20000000,
    )
    parser.add_argument(
        "--save",
        help="save aligned second trajectory (format: stamp x y z)",
    )
    parser.add_argument(
        "--save_associations",
        help="save associated trajectories",
    )
    parser.add_argument(
        "--plot",
        help="plot trajectories to image (e.g. result.png)",
    )
    parser.add_argument(
        "--verbose",
        help="print detailed evaluation statistics",
        action="store_true",
    )
    parser.add_argument(
        "--verbose2",
        help="print RMSE with and without scale correction",
        action="store_true",
    )
    args = parser.parse_args()

    first_list = associate.read_file_list(args.first_file, False)
    second_list = associate.read_file_list(args.second_file, False)

    matches = associate.associate(
        first_list,
        second_list,
        float(args.offset),
        float(args.max_difference),
    )
    if len(matches) < 2:
        sys.exit(
            "Couldn't find matching timestamp pairs between groundtruth "
            "and estimated trajectory! Did you choose the correct sequence?"
        )

    first_xyz = np.matrix(
        [[float(value) for value in first_list[a][0:3]] for a, b in matches]
    ).transpose()
    second_xyz = np.matrix(
        [
            [float(value) * float(args.scale) for value in second_list[b][0:3]]
            for a, b in matches
        ]
    ).transpose()

    sorted_second_list = sorted(second_list.items())
    second_xyz_full = np.matrix(
        [
            [float(value) * float(args.scale) for value in sorted_second_list[i][1][0:3]]
            for i in range(len(sorted_second_list))
        ]
    ).transpose()

    rot, transGT, trans_errorGT, trans, trans_error, scale = align(
        second_xyz, first_xyz
    )

    second_xyz_aligned = scale * rot * second_xyz + trans
    second_xyz_notscaled_full = rot * second_xyz_full + trans

    first_stamps = sorted(first_list.keys())
    first_xyz_full = _to_xyz_array(first_list, first_stamps)

    second_stamps = sorted(second_list.keys())
    second_xyz_full_plot = _to_xyz_array(second_list, second_stamps, args.scale)
    second_xyz_full_aligned = scale * rot * second_xyz_full_plot + trans

    rmse = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
    rmse_gt = np.sqrt(np.dot(trans_errorGT, trans_errorGT) / len(trans_errorGT))

    if args.verbose:
        print("compared_pose_pairs {} pairs".format(len(trans_error)))
        print("absolute_translational_error.rmse {:.6f} m".format(rmse))
        print("absolute_translational_error.mean {:.6f} m".format(np.mean(trans_error)))
        print("absolute_translational_error.median {:.6f} m".format(np.median(trans_error)))
        print("absolute_translational_error.std {:.6f} m".format(np.std(trans_error)))
        print("absolute_translational_error.min {:.6f} m".format(np.min(trans_error)))
        print("absolute_translational_error.max {:.6f} m".format(np.max(trans_error)))
        print("max idx: {}".format(int(np.argmax(trans_error))))
        print("scale {:.6f}".format(scale))
    else:
        print("{:.6f},{:.6f},{:.6f}".format(rmse, scale, rmse_gt))

    if args.verbose2:
        print("compared_pose_pairs {} pairs".format(len(trans_error)))
        print("absolute_translational_error.rmse {:.6f} m".format(rmse))
        print("absolute_translational_errorGT.rmse {:.6f} m".format(rmse_gt))

    if args.save_associations:
        with open(args.save_associations, "w") as f:
            lines = [
                "{:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(
                    a, x1, y1, z1, b, x2, y2, z2
                )
                for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(
                    matches,
                    np.asarray(first_xyz.transpose()).reshape(-1, 3),
                    np.asarray(second_xyz_aligned.transpose()).reshape(-1, 3),
                )
            ]
            f.write("\n".join(lines))

    if args.save:
        with open(args.save, "w") as f:
            lines = [
                "{:f} {:f} {:f} {:f}".format(
                    stamp, line[0], line[1], line[2]
                )
                for stamp, line in zip(
                    second_stamps,
                    np.asarray(second_xyz_notscaled_full.transpose()).reshape(-1, 3),
                )
            ]
            f.write("\n".join(lines))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_traj(
            ax,
            first_stamps,
            np.asarray(first_xyz_full.transpose()).reshape(-1, 3),
            "-",
            "black",
            "ground truth",
        )
        plot_traj(
            ax,
            second_stamps,
            np.asarray(second_xyz_full_aligned.transpose()).reshape(-1, 3),
            "-",
            "blue",
            "estimated",
        )
        label = "difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(
            matches,
            np.asarray(first_xyz.transpose()).reshape(-1, 3),
            np.asarray(second_xyz_aligned.transpose()).reshape(-1, 3),
        ):
            ax.plot([x1, x2], [y1, y2], "-", color="red", label=label)
            label = ""

        ax.legend()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.axis("equal")
        plt.savefig(args.plot, format="pdf")
        print("Saved plot to {}".format(args.plot))
