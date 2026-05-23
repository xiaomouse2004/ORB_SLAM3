# FastGS 数据转换脚本

## EuRoC + ORB-SLAM3 → COLMAP（FastGS）

将 EuRoC 格式数据集（仅 **cam0**）与 ORB-SLAM3 输出的相机轨迹，转换为 FastGS 可读取的 **COLMAP** 目录结构。

### 输入要求

**EuRoC 目录**（示例）：

```
euroc_VID_Arealm_20260417160055790_raw/
├── mav0/
│   ├── cam0/
│   │   ├── data/              # RGB PNG
│   │   ├── data.csv
│   │   └── sensor.yaml
│   └── depth/
│       ├── data/              # 16-bit 深度 PNG（与 cam0 同文件名）
│       └── depth_meta.json    # 每帧 d_min / d_max
├── depth_readme.md            # 深度解码说明
└── CameraTrajectory.txt       # ORB-SLAM3 输出（推荐，稠密）
```

**ORB-SLAM3 轨迹格式**（TUM，默认）：

```
# timestamp tx ty tz qx qy qz qw
77004585986000 0.0 0.0 0.0 0.0 0.0 0.0 1.0
```

也支持 EuRoC CSV 格式（`qw qx qy qz` 顺序）。

> 说明：ORB-SLAM3 保存的是 **T_wc**（相机在世界系下的位姿）。脚本会自动转换为 COLMAP 的 **world-to-camera** 外参。  
> **初始彩色点云**：默认用 `mav0/depth` 反投影（深度 **0.3–10 m**，颜色来自 cam0 RGB）；无深度时回退为沿光轴合成灰点。

### 依赖

```bash
conda activate fastgs
pip install pyyaml
```

### 转换命令

**你的数据集示例：**

```bash
cd /home/jackson/opensource/FastGS

python scripts/euroc_orbslam3_to_colmap.py \
  --euroc-root /home/jackson/data/euroc/euroc_VID_Arealm_20260417160055790_raw \
  --output-dir /home/jackson/opensource/FastGS/data/euroc_arealm_cam0 \
  --depth-min 0.3 \
  --depth-max 10.0 \
  --clean
```

ORB-SLAM3 轨迹默认读取 **EuRoC 根目录** 下的 `CameraTrajectory.txt`（不存在则尝试 `KeyFrameTrajectory.txt`）。

### 常用参数

| 参数 | 说明 |
|------|------|
| `--trajectory-format auto\|tum\|euroc` | 轨迹格式，默认 `auto` |
| `--max-time-delta-ns 50000000` | 图像与轨迹时间戳最大允许偏差（50ms） |
| `--copy-images` | 复制图像（默认创建符号链接，省磁盘） |
| `--max-images 200` | 仅转换前 N 帧（调试） |
| `--num-init-points 50000` | 初始 3D 点数量（下采样后） |
| `--depth-min 0.3` / `--depth-max 10.0` | 有效深度范围（米） |
| `--depth-pixel-stride 8` | 深度反投影像素步长（越大越快、点越少） |
| `--depth-frame-step 1` | 每隔 N 帧取一帧深度 |
| `--no-use-depth` | 禁用深度，使用灰点合成初始化 |

### 输出结构

```
data/euroc_arealm_cam0/
├── images/                 # cam0 图像（链接或拷贝）
└── sparse/0/
    ├── cameras.txt
    ├── images.txt
    ├── points3D.txt
    └── points3D.ply
```

### FastGS 训练

```bash
conda activate fastgs
cd /home/jackson/opensource/FastGS

# 自定义序列建议不加 --eval（无标准 test split）
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=euroc_arealm python train.py \
  -s /home/jackson/opensource/FastGS/data/euroc_arealm_cam0 \
  -i images \
  --test_iterations 30000 \
  --densification_interval 500
```

渲染与可视化：

```bash
python render.py -m output/euroc_arealm --skip_train
# 3D 预览：output/euroc_arealm/point_cloud/iteration_30000/point_cloud.ply
# 上传至 https://superspl.at/editor
```

### 常见问题

1. **`No cam0 frames matched trajectory timestamps`**  
   - 确认轨迹时间戳与 `data.csv` 同为 **纳秒**，或同为秒（脚本会自动尝试对齐）。  
   - 优先使用 `CameraTrajectory.txt`（每帧/稠密），而非仅关键帧的 `KeyFrameTrajectory.txt`。  
   - 增大 `--max-time-delta-ns`（如 `100000000` = 100ms）。

2. **图像分辨率 1920×1600，显存不足**  
   训练时加降采样：`-r 2` 或 `-r 4`。

3. **需要拷贝图像而非符号链接**  
   加 `--copy-images`（约 717 帧 × ~1MB/帧，请预留磁盘空间）。
