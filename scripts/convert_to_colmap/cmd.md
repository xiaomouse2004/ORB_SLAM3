# FastGS：EuRoC 数据转换与训练命令

修改下面 **变量定义** 即可适配不同序列；后续命令均引用这些变量。

---

## 变量定义

**推荐：** 只改 `scripts/env.sh` 里的路径，然后在终端加载：

```bash
source /home/jackson/opensource/FastGS/scripts/env.sh
```

或进入仓库后：

```bash
source scripts/env.sh
```

`env.sh` 中的变量说明：

```bash
FASTGS_ROOT      # FastGS 工程根目录
EUROC_ROOT       # EuRoC 原始数据根（含 mav0/、CameraTrajectory.txt）
EUROC_NAME       # 序列名（自动取 EUROC_ROOT 目录名）
COLMAP_DATA      # 转换输出：${FASTGS_ROOT}/data/${EUROC_NAME}_cam0
MODEL_OUTPUT     # 训练输出：${FASTGS_ROOT}/output/${EUROC_NAME}
TRAJECTORY_FILE  # ${EUROC_ROOT}/CameraTrajectory.txt
DEPTH_MIN        # 深度下限（米），默认 0.3
DEPTH_MAX        # 深度上限（米），默认 10.0
CUDA_VISIBLE_DEVICES
```

---

## 路径一览

| 变量 | 含义 | 当前示例值 |
|------|------|------------|
| `FASTGS_ROOT` | FastGS 仓库 | `/home/jackson/opensource/FastGS` |
| `EUROC_ROOT` | EuRoC + ORB-SLAM3 数据根 | `.../euroc_VID_Arealm_20260417160341422` |
| `EUROC_NAME` | 序列名 | `euroc_VID_Arealm_20260417160341422` |
| `COLMAP_DATA` | 转换后的 COLMAP 数据 | `${FASTGS_ROOT}/data/${EUROC_NAME}_cam0` |
| `MODEL_OUTPUT` | 训练输出 | `${FASTGS_ROOT}/output/${EUROC_NAME}` |
| `TRAJECTORY_FILE` | 相机轨迹 | `${EUROC_ROOT}/CameraTrajectory.txt` |

---

## 1. 数据转换

```bash
source scripts/env.sh   # 或 source /home/jackson/opensource/FastGS/scripts/env.sh
```

**前提：**

- `${TRAJECTORY_FILE}` 存在
- `${EUROC_ROOT}/mav0/cam0/`、`mav0/depth/data/`、`mav0/depth/depth_meta.json` 齐全

```bash
conda activate fastgs
cd "${FASTGS_ROOT}"

python scripts/euroc_orbslam3_to_colmap.py \
  --euroc-root "${EUROC_ROOT}" \
  --output-dir "${COLMAP_DATA}" \
  --depth-min "${DEPTH_MIN}" \
  --depth-max "${DEPTH_MAX}" \
  --clean
```

---

## 2. 训练（16GB 显存推荐）

```bash
source scripts/env.sh
conda activate fastgs
cd "${FASTGS_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
OAR_JOB_ID="${EUROC_NAME}" \
python train.py \
  -s "${COLMAP_DATA}" \
  -i images \
  -r 2 \
  --test_iterations 30000 \
  --densification_interval 500
```

训练完成后模型在：`${MODEL_OUTPUT}/`

---

## 3. 评测（可选）

```bash
source scripts/env.sh
conda activate fastgs
cd "${FASTGS_ROOT}"

# 训练时加 --eval
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
OAR_JOB_ID="${EUROC_NAME}" \
python train.py \
  -s "${COLMAP_DATA}" \
  -i images \
  -r 2 \
  --eval \
  --test_iterations 30000 \
  --densification_interval 500

# 训练结束后渲染与指标
python render.py -m "${MODEL_OUTPUT}" --skip_train --iteration -1
python metrics.py -m "${MODEL_OUTPUT}"
```
