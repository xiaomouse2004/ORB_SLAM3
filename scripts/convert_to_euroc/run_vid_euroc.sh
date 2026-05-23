#!/usr/bin/env bash
# Usage: ./run_vid_euroc.sh INPUT_ROOT OUTPUT_ROOT DATA_ID
#
# Example:
#   ./run_vid_euroc.sh \
#     /root/workspace/ORB_SLAM3/data/3drecon_raw \
#     /root/workspace/ORB_SLAM3/data \
#     VID_Arealm_20260417160055790

set -e

input_root="$1"
output_root="$2"
data_id="$3"

ORB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
input_dir="${input_root}/${data_id}"
output_dir="${output_root}/euroc_${data_id}"

# python3 "${ORB_ROOT}/scripts/convert_3drecon_to_euroc.py" \
#   --input-dir "${input_dir}" \
#   --output-dir "${output_dir}" \
#   --undistorted

cd "${ORB_ROOT}"

# # Run stereo_inertial_euroc
# ./Examples/Stereo-Inertial/stereo_inertial_euroc \
#   ./Vocabulary/ORBvoc.txt \
#   "${output_dir}/ORB_SLAM3_stereo_inertial_settings.yaml" \
#   "${output_dir}" \
#   "${output_dir}/stereo_timestamps_ns.txt"

# # Run mono_inertial_euroc
# ./Examples/Monocular-Inertial/mono_inertial_euroc \
#   ./Vocabulary/ORBvoc.txt \
#   "${output_dir}/ORB_SLAM3_mono_inertial_settings.yaml" \
#   "${output_dir}" \
#   "${output_dir}/mono_timestamps_ns.txt"

# Run stereo_euroc
./Examples/Stereo/stereo_euroc \
  ./Vocabulary/ORBvoc.txt \
  "${output_dir}/ORB_SLAM3_stereo_settings.yaml" \
  "${output_dir}" \
  "${output_dir}/stereo_timestamps_ns.txt"