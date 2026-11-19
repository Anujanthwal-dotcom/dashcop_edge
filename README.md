# DashCop Edge: Real-Time Traffic Violation Detection

An online, edge-optimized reimplementation of **DashCop** (*WACV 2025: Automated E-ticket Generation for Two-Wheeler Traffic Violations Using Dashcam Videos*).

## Key Features
- **Real-Time Streaming Pipeline**: Single-pass, zero-latency ingestion with automatic frame dropping under queue congestion.
- **Dynamic Ingestion Toggle**: Seamlessly switch between **Rate-Paced Video Simulation** (`--mode video`) and **Live USB Camera** (`--mode usb`).
- **Fast Online CA-SORT**: Replaces offline Gurobi ILP solvers with real-time Kalman filtering and bipartite matching ($< 1.5\text{ ms}$ latency).
- **Event-Driven Best-Shot ANPR**: Automatically selects the sharpest vehicle frame for single-shot license plate OCR, avoiding heavy full-frame transformer inference every frame.
- **Drop-in ONNX Architecture**: Modular inference engine configured for ONNX Runtime (CPU / CUDA / TensorRT).

---

## Getting Started with `uv`

### 1. Install Dependencies (when you are ready)
```bash
# Sync all dependencies into a virtual environment
uv sync

# Or with GPU acceleration
uv sync --extra gpu
```

### 2. Add ONNX Model Weights
Place your trained `.onnx` models into the `weights/` directory:
- `weights/sac.onnx`
- `weights/hnh.onnx`
- `weights/lp_det.onnx`
- `weights/lp_ocr.onnx`

*(All 4 models are strictly required. If any file is missing, the system throws a descriptive error and stops).*

> **Train in the Cloud for Free (Google Colab & Kaggle):**
> You can train all 4 models directly in the cloud without downloading the dataset to your local PC using either:
> - **Google Colab:** [train_on_colab.ipynb](file:///home/anuj/Desktop/DashCop_Recreated/train_on_colab.ipynb) (Free Tesla T4 GPU)
> - **Kaggle Notebooks:** [train_on_kaggle.ipynb](file:///home/anuj/Desktop/DashCop_Recreated/train_on_kaggle.ipynb) (Free GPU P100 / Dual T4, 30 hrs/week)
> Both notebooks automatically download the complete RideSafe-400 dataset in the cloud, train all 4 models, and package `dashcop_weights.zip` for download.

### 3. Run the System

#### Mode A: Real-Time Video File Simulation
Rate-paces video playback to match camera recording FPS:
```bash
uv run dashcop-edge --mode video --source data/sample.mp4 --display
```

#### Mode B: Live USB Dashcam
Connects to a live V4L2 USB camera:
```bash
uv run dashcop-edge --mode usb --source 0 --display
```

---

## Project Structure
```
DashCop_Recreated/
├── pyproject.toml              # uv project configuration & dependencies
├── .python-version             # Python version pin
├── configs/
│   └── edge_config.yaml        # Runtime thresholds, buffers & paths
├── src/
│   └── dashcop_edge/           # Real-time modular package
├── weights/
│   └── README.md               # Drop-in ONNX model specifications
├── output/
│   └── violations/             # Generated E-Ticket JSON records & evidence crops
└── temp/                       # Upstream offline reference repository
```
