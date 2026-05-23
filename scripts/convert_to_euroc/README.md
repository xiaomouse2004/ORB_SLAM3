# ORB-SLAM3 辅助脚本

## 依赖

- `ffmpeg`（用于从并排立体 MP4 中裁切左右目 PNG）
- Python 3：`numpy`、`PyYAML`
- 轨迹绘图脚本另需：`matplotlib`（与 `numpy`）

在 Ubuntu / Debian 上可安装：

```bash
apt-get update
apt-get install -y ffmpeg python3-numpy python3-yaml python3-matplotlib
```

## `plot_camera_trajectory.py`

将 ORB-SLAM3 **`SaveTrajectoryEuRoC`** 输出的轨迹（如 `CameraTrajectory.txt`、`f_*.txt`）画成 **XY / XZ / YZ** 与 **3D** 路径图。每行格式：`timestamp(ns) tx ty tz qx qy qz qw`。

示例（工程根目录下的默认轨迹名）：

```bash
python3 /root/workspace/ORB_SLAM3/scripts/plot_camera_trajectory.py \
  -i /root/workspace/ORB_SLAM3/CameraTrajectory.txt \
  -o /root/workspace/ORB_SLAM3/camera_trajectory_plot.png
```

- `-i`：输入轨迹文件路径；省略时默认读取**当前工作目录**下的 `CameraTrajectory.txt`。
- `-o`：输出 PNG 路径；省略时默认为 `trajectory_plot.png`。
- 仅 2D 三视图、不要 3D 子图时加 `--no-3d`；分辨率用 `--dpi`（默认 150）。

若未安装 matplotlib：`pip install matplotlib numpy` 或使用上面的 `apt-get install python3-matplotlib`。

## `convert_3drecon_to_euroc.py`

将 `3drecon` 目录下的 **VID_*** 结果（左右并排 SBS 视频、`cam_ts.csv`、`camchain-imucam.yaml`、可选 `imu_raw.csv`）转换为 **EuRoC ASL** 风格的 `mav0/` 目录树，便于使用 `stereo_euroc`、`mono_euroc` 等程序。

### 输入目录需包含

| 文件 | 说明 |
|------|------|
| `<stem>.mp4` | 左右目水平并排的立体视频 |
| `<stem>_cam_ts.csv` | 表头含 `frame_index,timestamp_ns`（帧号从 0 连续递增） |
| `camchain-imucam.yaml` | Kalibr 风格标定（cam0 / cam1 内参、`T_cn_cnm1` 等） |
| `imu.yaml` | 可选；Kalibr IMU 噪声 / `update_rate` / `time_offset`（`imu0:` 块），用于 `ORB_SLAM3_stereo_inertial_settings.yaml` |
| `<stem>_imu_raw.csv` | 可选；含 `timestamp_ns,gyro_*,acc_*`，将写入 `mav0/imu0/data.csv`（EuRoC 列名） |
| `depth_imgs/` | 可选；`{frame_index}.png` 16-bit 深度图，按 `*_cam_ts.csv` 重命名为 `mav0/depth/data/<timestamp_ns>.png` |
| `depth_imgs/depth_meta.json` | 可选；输出到 `mav0/depth/depth_meta.json`，以 `timestamp_ns`（字符串）为 key 的对象 |

### 一键脚本 `run_vid_euroc.sh`

三个参数：`input_root`、`output_root`、`data_id`。依次执行 **VID→EuRoC 转换** 与 **双目+IMU**。输出目录为 **`${output_root}/euroc_${data_id}_raw`**。

```bash
cd /root/workspace/ORB_SLAM3/scripts
chmod +x run_vid_euroc.sh

./run_vid_euroc.sh \
  /root/workspace/ORB_SLAM3/data/3drecon_raw \
  /root/workspace/ORB_SLAM3/data \
  VID_Arealm_20260417160055790
```

### 路径变量（手动命令）

在 shell 里先设好变量，**只改 `data_id` 和根路径**即可复用下面所有命令：

| 变量 | 含义 | 示例 |
|------|------|------|
| `ORB_ROOT` | ORB-SLAM3 工程根目录 | `/root/workspace/ORB_SLAM3` |
| `input_root` | 原始 `3drecon` / `3drecon_raw` 数据根目录 | `.../data/3drecon_raw` |
| `output_root` | EuRoC 转换结果根目录 | `.../data` |
| `data_id` | VID 包目录名（= `<stem>`） | `VID_Arealm_20260417160055790` |
| `input_dir` | `${input_root}/${data_id}` | 含 mp4、cam_ts、depth_imgs 等 |
| `output_dir` | `${output_root}/euroc_${data_id}_raw` | 含 `mav0/`、ORB yaml |

```bash
ORB_ROOT=/root/workspace/ORB_SLAM3
input_root="${ORB_ROOT}/data/3drecon_raw"
output_root="${ORB_ROOT}/data"

data_id=VID_Arealm_20260417160055790

input_dir="${input_root}/${data_id}"
output_dir="${output_root}/euroc_${data_id}_raw"
```

数据在 AutoDL 磁盘时，可改为：

```bash
ORB_ROOT=/root/workspace/ORB_SLAM3
input_root=/root/autodl-tmp/data/3drecon_raw
output_root=/root/autodl-tmp/data
data_id=VID_Arealm_20260417160055790
input_dir="${input_root}/${data_id}"
output_dir="${output_root}/euroc_${data_id}_raw"
```

转换前检查输入是否齐全：

```bash
ls "${input_dir}/${data_id}.mp4" \
   "${input_dir}/${data_id}_cam_ts.csv" \
   "${input_dir}/camchain-imucam.yaml" \
   "${input_dir}/depth_imgs"
```

### 转换命令（VID → EuRoC）

```bash
python3 "${ORB_ROOT}/scripts/convert_3drecon_to_euroc.py" \
  --input-dir "${input_dir}" \
  --output-dir "${output_dir}" \
  --undistorted
```

若 MP4 里的画面**已经过去畸变**（或立体校正），请保留 **`--undistorted`**：脚本会把 `sensor.yaml` 与 ORB 配置中的 **`k1,k2,p1,p2` 全部写成 0**（`0.0` 形式，兼容 ORB-SLAM3 `readParameter<float>`）。此时 `camchain-imucam.yaml` 的 **内参与分辨率必须与去畸变后的图像一致**。

其他常用选项：`--keep-temp`（保留 ffmpeg 临时帧）、`--no-depth`（不处理 `depth_imgs/`）、`--depth-imgs-dir PATH`（指定深度图目录）。

### 输出说明

在 `--output-dir` 下生成：

- `mav0/cam0/data/<timestamp_ns>.png`、`mav0/cam1/data/<timestamp_ns>.png`：与 EuRoC 相同的命名规则
- `mav0/cam0/data.csv`、`mav0/cam1/data.csv`：EuRoC 风格索引
- `mav0/depth/data/<timestamp_ns>.png`、`mav0/depth/data.csv`、`mav0/depth/depth_meta.json`：若存在 `depth_imgs/` 则自动生成；`depth_meta.json` 形如 `{"77004585986000": {"frame": 0, "d_min": ..., "d_max": ...}, ...}`
- `mav0/cam0/sensor.yaml`、`mav0/cam1/sensor.yaml`：由 `camchain-imucam.yaml` 推导（`T_BS = inv(T_cam_imu)`，body 取 IMU）
- `mav0/body.yaml`：占位说明
- `mav0/imu0/data.csv`：若存在 `*_imu_raw.csv`，仅保留首末相机时间戳前后各 **0.5 s**（可用 `--imu-margin-ns` 调整）内的 IMU 样本
- `mav0/imu0/sensor.yaml`：EuRoC 风格 IMU 标定（`rate_hz`、噪声密度、`T_BS`；参数来自 `imu.yaml` 或 EuRoC 默认）
- `stereo_timestamps_ns.txt`：每行一个纳秒时间戳，供 ORB-SLAM3 `stereo_euroc` 使用（与官方 `EuRoC_TimeStamps/MH01.txt` 格式相同）
- `ORB_SLAM3_stereo_settings.yaml`：双目 ORB 配置（`Camera.fps` 为**整数**）
- `ORB_SLAM3_mono_settings.yaml`：单目（cam0 / 左目）ORB 配置，`Camera.RGB: 0` 与 `mono_euroc` 读入的 OpenCV BGR 一致；畸变系数格式与双目相同（零畸变时为 `0.0`）
- `ORB_SLAM3_stereo_inertial_settings.yaml`：双目+IMU（存在 `*_imu_raw.csv` 且可解析 `imu.yaml` 时）；`IMU.Noise*` 来自 `imu.yaml`，`IMU.Frequency` 为 `update_rate`；IMU 时间戳会加上 `camchain timeshift_cam_imu + imu.yaml time_offset`

调试时可保留 ffmpeg 临时帧目录：`--keep-temp`。

### 运行 ORB-SLAM3（转换完成后）

`stereo_euroc` / `mono_euroc` / `stereo_inertial_euroc` 的**序列根目录**为 `output_dir`（包含 `mav0/` 的父路径）。请先 `cd "${ORB_ROOT}"`，并确保已编译对应可执行文件。

**双目 + IMU（推荐）**

在 `input_dir` 放置 Kalibr 风格 **`imu.yaml`** 与 `<data_id>_imu_raw.csv` 时，转换会生成 `ORB_SLAM3_stereo_inertial_settings.yaml`：

| `imu.yaml` 字段 | 写入 ORB-SLAM3 |
|-----------------|----------------|
| `gyroscope_noise_density` | `IMU.NoiseGyro` |
| `accelerometer_noise_density` | `IMU.NoiseAcc` |
| `gyroscope_random_walk` | `IMU.GyroWalk` |
| `accelerometer_random_walk` | `IMU.AccWalk` |
| `update_rate` | `IMU.Frequency` |
| `time_offset` + `cam0.timeshift_cam_imu` | 写入 `mav0/imu0/data.csv` 时平移时间戳（纳秒） |

```bash
cd "${ORB_ROOT}"

./Examples/Stereo-Inertial/stereo_inertial_euroc \
  ./Vocabulary/ORBvoc.txt \
  "${output_dir}/ORB_SLAM3_stereo_inertial_settings.yaml" \
  "${output_dir}" \
  "${output_dir}/stereo_timestamps_ns.txt"
```

**仅双目**

```bash
cd "${ORB_ROOT}"

./Examples/Stereo/stereo_euroc \
  ./Vocabulary/ORBvoc.txt \
  "${output_dir}/ORB_SLAM3_stereo_settings.yaml" \
  "${output_dir}" \
  "${output_dir}/stereo_timestamps_ns.txt" \
  trajectory_${data_id}
```

**仅单目（cam0 / 左目）**

```bash
cd "${ORB_ROOT}"

./Examples/Monocular/mono_euroc \
  ./Vocabulary/ORBvoc.txt \
  "${output_dir}/ORB_SLAM3_mono_settings.yaml" \
  "${output_dir}" \
  "${output_dir}/stereo_timestamps_ns.txt"
```

说明：

- 第二个参数请使用转换生成的 **`ORB_SLAM3_*_settings.yaml`**，不要使用官方 `Examples/Stereo/EuRoC.yaml`（分辨率与内参与 VID 设备不一致）。
- `stereo_euroc` 最后一个参数为可选轨迹文件名前缀；省略时默认 `CameraTrajectory.txt` / `KeyFrameTrajectory.txt`。
- 无 `imu.yaml` 时仍可用 EuRoC 默认 IMU 噪声；时间对齐仅应用 `camchain` 的 `timeshift_cam_imu`（若有）。
- 无 Pangolin 显示时需在 yaml 或工程中关闭 Viewer。

### 官方 EuRoC MH01（对照）

结构与 VID 相同：`DATA_ROOT` 下为 `mav0/`，时间戳用官方列表：

```bash
ORB_ROOT=/root/workspace/ORB_SLAM3
DATA_ROOT="${ORB_ROOT}/data"   # 含 mav0/cam0、cam1、imu0

cd "${ORB_ROOT}"

./Examples/Stereo-Inertial/stereo_inertial_euroc \
  ./Vocabulary/ORBvoc.txt \
  ./Examples/Stereo-Inertial/EuRoC.yaml \
  "${DATA_ROOT}" \
  ./Examples/Stereo-Inertial/EuRoC_TimeStamps/MH01.txt
```
