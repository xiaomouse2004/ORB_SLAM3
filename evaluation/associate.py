#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Associate timestamps between two trajectory files.
"""

import argparse
import sys


def read_file_list(filename, remove_bounds=False):
    """
    Reads a trajectory from a text file.

    File format: stamp d1 d2 d3 ...
    """
    with open(filename) as f:
        data = f.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    if remove_bounds:
        lines = lines[100:-100]
    entries = [
        [v.strip() for v in line.split(" ") if v.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    entries = [(float(row[0]), row[1:]) for row in entries if len(row) > 1]
    return dict(entries)


def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp, data) by closest timestamp.
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Associate two timestamped data files."
    )
    parser.add_argument("first_file", help="first text file (format: timestamp data)")
    parser.add_argument("second_file", help="second text file (format: timestamp data)")
    parser.add_argument(
        "--first_only",
        help="only output associated lines from first file",
        action="store_true",
    )
    parser.add_argument(
        "--offset",
        help="time offset added to timestamps of the second file (default: 0.0)",
        default=0.0,
    )
    parser.add_argument(
        "--max_difference",
        help="max allowed time difference for matching (default: 0.02)",
        default=0.02,
    )
    args = parser.parse_args()

    first_list = read_file_list(args.first_file)
    second_list = read_file_list(args.second_file)

    matches = associate(
        first_list, second_list, float(args.offset), float(args.max_difference)
    )

    if args.first_only:
        for a, b in matches:
            print("{} {}".format(a, " ".join(first_list[a])))
    else:
        for a, b in matches:
            print(
                "{} {} {} {}".format(
                    a,
                    " ".join(first_list[a]),
                    b - float(args.offset),
                    " ".join(second_list[b]),
                )
            )
