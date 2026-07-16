# track_Motion · Microsphere Detection & Motion Tracking



---

## ✨ Features

- **Enhanced YOLO detection**
  - Color-attention module (`ColorAttentionModule` / `HSVAttention`): strengthens perception of differently colored microspheres (red / blue / white / black).
  - Scale-aware FPN (`ScaleAwareFPN`): improves detection of microspheres across sizes.
  - Multi-task single model: bounding boxes + color classification + size regression, trained with `EnhancedDetectionLoss` (detection + color + size).
- **Motion tracking**
  - Custom `MicroBeadTracker`: position / class / direction / angle based matching, with disappear & appear zones, a motion model (velocity / direction smoothing), lost-track recovery, and statistics.
  - Optional `ByteTrack` via `bytetrack_tracker.py`.
  - Visualization: tracked video, matching-area video, trajectory plot (PNG), and a results CSV (displacement & velocity).
- **Flexible entry points**
  - Training (`train.py`), detection inference (`predict.py`), single-video tracking (`Tracker.py`), batch video tracking (`Mutil_tracker.py`).
- **Engineering niceties**
  - Centralized config (`config.py`) overridable from the CLI.
  - Inference supports checkpoint resume, sliding-window for large images, and batch image/video/directory processing.
  - Auto-detects dataset structure and generates a temp `data.yaml`; supports multi-scale / rectangular training.

---



## 📁 Project Structure

```
track_Motion/
├── config.py                  # Central config (train/val/predict/microsphere/postprocess)
├── train.py                   # Training: loads enhanced model, combines detection/color/size losses
├── predict.py                 # Detection inference: image/video/dir batch, sliding window, checkpoint resume
├── Tracker.py                 # Single-video motion-tracking executor (wraps YOLOTrackPipeline)
├── Mutil_tracker.py           # Batch video tracking: recursive search, skip existing, summary/error logs
├── models/
│   └── enhanced_yolo.py       # Enhanced YOLO: ColorAttention + ScaleAwareFPN, multi-task outputs
├── utils/
│   ├── color_attention.py     # Color-attention module (ColorAttention / HSVAttention)
│   ├── scale_fpn.py           # Scale-aware FPN + size-regression loss SizeLoss
│   ├── microsphere_utils.py   # MicrosphereFeaturesExtractor
│   ├── data_augmentation.py   # Microsphere augmentation (preserves color/size: rotate/brightness/contrast/blur/noise/flip)
│   ├── process_utils.py       # process_image / process_image_sliding_window / process_video
│   ├── general.py             # ensure_dir / save_results_to_csv / create_plots, etc.
│   ├── cli_utils.py           # CLI argument parsing
│   ├── file_utils.py          # file helpers
│   └── __init__.py            # detect_dataset_structure / detect_class_names / split_dataset
├── Motion_tracker/
│   ├── tracker.py             # MicroBeadTracker (custom tracker) + YOLOTrackPipeline (detect+track pipeline)
│   ├── track.py               # Track: single-trajectory state
│   ├── matching.py            # MatchingStrategy (matching / disappear-zone / appear-zone)
│   ├── motion_model.py        # MotionModel / AdaptiveMotionModel
│   ├── data_association.py    # DataAssociation
│   ├── utils.py               # calculate_distance / calculate_angle / is_bbox_in_matching_region
│   ├── bytetrack_tracker.py   # ByteTrackWrapper
│   ├── visualization.py       # TrackVisualizer
│   ├── vis_match_area.py      # MatchingAreaVisualizer
│   └── __init__.py
├── weights/
│   └── yolov8l.pt             # YOLOv8 pretrained baseline (download separately, see Install)
└── data/                      # Dataset dir (YOLO format, add your own; gitignored)
```

---



## 🚀 Installation

Requires **Python 3.9+**.

```bash
# 1. Clone
git clone git@github.com:mortonhan/Motion_tracker.git
cd track_Motion

# 2. (Recommended) virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare baseline weights
# The enhanced model is based on YOLOv8. Place yolov8l.pt into weights/:
#   download from https://github.com/ultralytics/assets/releases (yolov8l.pt)
#   or run: yolo download model=yolov8l.pt
```

Core `requirements.txt`:

```
torch
ultralytics
opencv-python
numpy
tqdm
pyyaml
matplotlib
```

> Note: CUDA is optional. Without a GPU it falls back to CPU; Apple-silicon Macs auto-use MPS (`--device auto`).

---



## 🎯 Quick Start

### 1. Prepare the dataset

Put a YOLO-format dataset under `data/`, e.g.:

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
└── data.yaml               # nc / names / path
```

`utils/__init__.py` auto-detects the structure; a single set of images can also be auto-split into train/val.

### 2. Train

```bash
python train.py \
  --dataset data \
  --epochs 200 \
  --batch-size 16 \
  --img-size 512 \
  --use-enhanced-model --use-color-attention --use-scale-fpn \
  --device 0
```

Key switches:

- `--use-enhanced-model` enables the enhanced model; `--no-enhanced-model` reverts to base YOLOv8.
- `--multi-scale` + `--rect` help with large size variance / varied aspect ratios.
- Trained artifacts default to `runs/train/`.

### 3. Detection inference (predict)

```bash
python predict.py \
  --weights runs/train/<exp>/weights/best.pt \
  --source track_data/5type_antibiotics/0_1.mp4 \
  --img-size 1920 \
  --conf-thres 0.3 \
  --project prediction_results/4
```

- Supports single image, single video, and directory (recursive) batching; directory mode has **checkpoint resume** so it continues after interruption.
- `--sliding-window` enables sliding-window detection for very large images.
- Outputs: annotated images/video, MOT-format labels, CSV (with color & size), etc.

### 4. Single-video motion tracking

```bash
python Tracker.py \
  --video track_data/11111-2.mp4 \
  --weights runs/train/<exp>/weights/best.pt \
  --output tracking_results \
  --img-size 1920 \
  --tracker motion            # or bytetrack
```

Outputs (under `--output`):

- `<video>_tracked.mp4` — tracked video with trajectories
- `<video>_matching_areas.mp4` — matching-region visualization video
- `<video>_results.csv` — per-track ID / class / duration (frames) / displacement / velocity
- `<video>_trajectories.png` — trajectory plot

Common tracking params (see `Tracker.py`): `--disappear-zone-width`, `--appear-zone-width`, `--initial-match-range`, `--max-match-range`, `--max-lost-frames`, `--static-neighborhood`, etc.

### 5. Batch video tracking

```bash
python Mutil_tracker.py \
  --input track_data/5 \
  --weights runs/train/<exp>/weights/best.pt \
  --output tracking_results/5 \
  --recursive --pattern "*.mp4" \
  --skip-existing           # skip already-generated results
```

Produces `batch_processing_summary.txt` and (on failure) `batch_processing_errors.log`.



## 📄 License

See the `LICENSE` file for details. (Add your preferred license before publishing.)