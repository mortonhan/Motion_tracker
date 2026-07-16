# track_Motion · 微球检测与运动追踪系统

一个面向**微球（microsphere）目标**的计算机视觉项目：在 YOLOv8 基础上引入**颜色注意力机制**与**尺度感知 FPN**，实现微球的**检测、颜色识别、尺寸测量**，并配套一套**运动追踪（Motion Tracker）**与 **ByteTrack** 可选追踪器，能够基于视频连续跟踪微球轨迹、统计位移与速度。

> 适用场景：显微镜 / 工业相机下的微球运动分析、颗粒计数与轨迹可视化等。

---

## ✨ 主要功能

- **增强型 YOLO 检测**
  - 颜色注意力模块（`ColorAttentionModule` / `HSVAttention`）：强化对红、蓝、白、黑等不同颜色微球的感知。
  - 尺度感知 FPN（`ScaleAwareFPN`）：提升对不同尺寸微球的检测能力。
  - 单模型多任务输出：检测框 + 颜色分类 + 尺寸回归（`EnhancedDetectionLoss` 组合检测/颜色/尺寸三类损失）。
- **运动追踪**
  - 自定义 `MicroBeadTracker`：基于位置、类别、运动方向与角度的匹配策略，支持消失区 / 出现区、运动模型（速度/方向平滑）、轨迹丢失恢复与统计。
  - `ByteTrack` 可选：调用 `bytetrack_tracker.py` 进行常规多目标跟踪。
  - 可视化：跟踪视频、`匹配区域`视频、轨迹图（PNG）、结果 CSV（含位移与速度）。
- **灵活的执行入口**
  - 训练（`train.py`）、检测推理（`predict.py`）、单视频追踪（`Tracker.py`）、批量视频追踪（`Mutil_tracker.py`）。
- **工程化细节**
  - 配置集中化（`config.py`），支持命令行覆盖。
  - 推理支持断点续推（checkpoint）、滑动窗口大图检测、图片/视频/目录批量处理。
  - 自动探测数据集结构并生成临时 `data.yaml`，支持多尺度 / 矩形训练。

---

## 📁 项目结构

```
track_Motion/
├── config.py                  # 中心化配置（训练/验证/预测/微球参数/后处理）
├── train.py                   # 训练脚本：加载增强模型，组合检测/颜色/尺寸损失
├── predict.py                 # 检测推理：图片/视频/目录批量，滑动窗口，断点续推
├── Tracker.py                 # 单视频运动追踪执行器（封装 YOLOTrackPipeline）
├── Mutil_tracker.py           # 批量视频追踪：递归搜索、跳过已处理、生成汇总/错误日志
├── models/
│   └── enhanced_yolo.py       # 增强 YOLO：ColorAttention + ScaleAwareFPN，多任务输出
├── utils/
│   ├── color_attention.py     # 颜色注意力模块（ColorAttention / HSVAttention）
│   ├── scale_fpn.py           # 尺度感知 FPN + 尺寸回归损失 SizeLoss
│   ├── microsphere_utils.py   # 微球特征提取 MicrosphereFeaturesExtractor
│   ├── data_augmentation.py   # 微球数据增强（保持颜色/尺寸：旋转/亮度/对比度/模糊/噪声/翻转）
│   ├── process_utils.py       # process_image / process_image_sliding_window / process_video
│   ├── general.py             # ensure_dir / save_results_to_csv / create_plots 等工具
│   ├── cli_utils.py           # 命令行参数解析
│   ├── file_utils.py          # 文件工具
│   └── __init__.py            # detect_dataset_structure / detect_class_names / split_dataset
├── Motion_tracker/
│   ├── tracker.py             # MicroBeadTracker（自定义追踪器）+ YOLOTrackPipeline（检测+追踪流水线）
│   ├── track.py               # Track 单条轨迹状态
│   ├── matching.py            # MatchingStrategy 匹配/消失区/出现区策略
│   ├── motion_model.py        # MotionModel / AdaptiveMotionModel 运动模型
│   ├── data_association.py    # DataAssociation 数据关联
│   ├── utils.py               # calculate_distance / calculate_angle / is_bbox_in_matching_region
│   ├── bytetrack_tracker.py   # ByteTrackWrapper 封装
│   ├── visualization.py       # TrackVisualizer 轨迹可视化
│   ├── vis_match_area.py      # MatchingAreaVisualizer 匹配区域可视化
│   └── __init__.py
├── weights/
│   └── yolov8l.pt             # YOLOv8 预训练基线权重（需自行下载，见安装说明）
└── data/                      # 数据集目录（YOLO 格式，自行放置，已被 .gitignore 忽略）
```

---

## 🚀 安装

要求 **Python 3.9+**。

```bash
# 1. 克隆仓库
git clone git@github.com:mortonhan/Motion_tracker.git
cd track_Motion

# 2. 创建并激活虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备基线权重
# 增强模型基于 YOLOv8，请将 yolov8l.pt 下载到 weights/ 目录：
#   https://github.com/ultralytics/assets/releases  （选择 yolov8l.pt）
# 或运行： yolo download model=yolov8l.pt
```

`requirements.txt` 依赖（核心）：

```
torch
ultralytics
opencv-python
numpy
tqdm
pyyaml
matplotlib
```

> 提示：CUDA 为可选。无 GPU 时会自动回退到 CPU；苹果芯片会自动使用 MPS（`--device auto`）。

---

## 🎯 快速开始

### 1. 准备数据集

将 YOLO 格式数据集放入 `data/`，结构示例：

```
data/
├── images/
│   ├── train/  *.jpg
│   ├── val/    *.jpg
│   └── test/   *.jpg
├── labels/
│   ├── train/  *.txt
│   ├── val/    *.txt
│   └── test/   *.txt
└── data.yaml               # nc / names / path 等
```

`utils/__init__.py` 会自动探测目录结构；若仅有一组图片也可自动切分生成训练/验证集。

### 2. 训练

```bash
python train.py \
  --dataset data \
  --epochs 200 \
  --batch-size 16 \
  --img-size 512 \
  --use-enhanced-model --use-color-attention --use-scale-fpn \
  --device 0
```

关键开关：
- `--use-enhanced-model` 启用增强模型；`--no-enhanced-model` 退回到基础 YOLOv8。
- `--multi-scale` + `--rect` 适合尺寸差异大/长宽比多样的图片。
- 训练产物默认保存到 `runs/train/`。

### 3. 检测推理（predict）

```bash
python predict.py \
  --weights runs/train/<exp>/weights/best.pt \
  --source track_data/5type_antibiotics/0_1.mp4 \
  --img-size 1920 \
  --conf-thres 0.3 \
  --project prediction_results/4
```

- 支持单图、单视频、目录（递归）批量；目录模式带 **checkpoint 断点续推**，中断后可继续。
- `--sliding-window` 可开启滑动窗口以处理超大图。
- 输出：带框图片/视频、MOT 格式标签、CSV（含颜色与尺寸）等。

### 4. 单视频运动追踪

```bash
python Tracker.py \
  --video track_data/11111-2.mp4 \
  --weights runs/train/<exp>/weights/best.pt \
  --output tracking_results \
  --img-size 1920 \
  --tracker motion            # 或 bytetrack
```

输出（位于 `--output`）：
- `<video>_tracked.mp4`：跟踪结果视频（含轨迹）
- `<video>_matching_areas.mp4`：匹配区域可视化视频
- `<video>_results.csv`：每条轨迹的 ID / 类别 / 持续帧数 / 位移 / 速度
- `<video>_trajectories.png`：轨迹图

常用追踪参数（见 `Tracker.py` 参数）：`--disappear-zone-width`、`--appear-zone-width`、`--initial-match-range`、`--max-match-range`、`--max-lost-frames`、`--static-neighborhood` 等。

### 5. 批量视频追踪

```bash
python Mutil_tracker.py \
  --input track_data/5 \
  --weights runs/train/<exp>/weights/best.pt \
  --output tracking_results/5 \
  --recursive --pattern "*.mp4" \
  --skip-existing           # 跳过已生成的结果
```

会输出 `batch_processing_summary.txt` 与（失败时的）`batch_processing_errors.log`。

---

## ⚙️ 配置说明

所有默认参数集中在 `config.py`，可在命令行用同名参数覆盖。主要块：

| 配置块 | 说明 |
| --- | --- |
| `MODEL_ENHANCEMENT` | 是否启用增强模型 / 颜色注意力 / 尺度 FPN |
| `DATA_AUGMENTATION` | 数据增强开关与范围（旋转、亮度、对比度、模糊、噪声、翻转） |
| `MODEL` | 基线模型名称（`yolov8l.pt`）、是否预训练 |
| `TRAIN` / `VAL` | 训练/验证超参（batch、epochs、img_size、优化器、学习率、多尺度范围等） |
| `PREDICT` | 推理默认参数、滑动窗口配置 |
| `MICROSPHERE` | 颜色 HSV 阈值、像素-微米换算比 `pixel_to_um_ratio`、尺寸测量开关 |
| `POSTPROCESS` | CSV 字段顺序、各类统计图表开关 |

> `MICROSPHERE.pixel_to_um_ratio` 用于将像素尺寸换算为物理尺寸（微米），请根据相机标定调整。

---

## 🧠 追踪算法简述

- **MicroBeadTracker（默认）**
  1. 将每帧 YOLO 检测转为统一的 `detection` 结构（bbox / 置信度 / 类别）。
  2. 过滤**消失区**（贴近左边缘的干扰区域）检测。
  3. 先对“静态检测”（与历史位置极近）做直接匹配，再对运动目标做**两轮匹配**（正常帧 + 丢失帧分别放宽匹配范围/角度）。
  4. 结合 `AdaptiveMotionModel` 预测下一帧位置，对丢失目标在更大范围内尝试恢复。
  5. 在**出现区**内未匹配的高置信检测作为新轨迹；输出连续轨迹、位移与速度统计。
- **ByteTrack**：标准 ByteTrack 关联（`track_thresh` / `track_buffer` / `match_thresh` 等），由 `bytetrack_tracker.py` 封装。

---

## 📊 输出产物

| 类型 | 路径 | 内容 |
| --- | --- | --- |
| 训练权重 | `runs/train/<exp>/weights/best.pt` | 训练得到的最佳模型 |
| 检测框/视频 | `prediction_results/<exp>/` | 图片、MOT 标签、CSV |
| 跟踪视频 | `tracking_results/` | `<video>_tracked.mp4`、`_matching_areas.mp4`、`_results.csv`、`_trajectories.png` |
| 批量报告 | `tracking_results/` | `batch_processing_summary.txt`、`batch_processing_errors.log` |

---

## 📄 许可证

详见 `LICENSE` 文件。（发布前请添加你选择的许可证。）
