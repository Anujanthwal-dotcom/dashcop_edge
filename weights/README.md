# DashCop Edge: ONNX Weights Directory

Place your trained `.onnx` models into this directory. All 4 model files are strictly required to run DashCop Edge.

> [!IMPORTANT]
> **Strict Model Existence Enforcement**:
> DashCop Edge will validate the presence of all required `.onnx` model files at startup. If any model file is missing, the system will immediately abort execution and print a detailed error indicating the missing file path.

---

## Required Model Files

### 1. `sac.onnx` (Segmentation & Cross-Association)
- **Path:** `weights/sac.onnx`
- **Architecture:** YOLOv8-Seg with Cross-Mask Head (WACV SAC module)
- **Input shape:** `(1, 3, 640, 640)` (FP32 or FP16)
- **Outputs:**
  - `boxes`: `(1, N, 6)` or bounding boxes `(xyxy, conf, cls)` where `cls: 0` = Rider, `cls: 1` = Motorbike
  - `masks`: Instance segmentation masks `(N, H, W)`
  - `cross_masks`: Cross-association masks linking riders to bikes `(N, H, W)`

### 2. `hnh.onnx` (Helmet / No-Helmet Detection)
- **Path:** `weights/hnh.onnx`
- **Architecture:** YOLOv8 detector / classifier
- **Input shape:** `(1, 3, 128, 128)` (rider head crop)
- **Classes:** `0` = No-Helmet (Violation), `1` = Helmet (Compliant)

### 3. `lp_det.onnx` (License Plate Detection)
- **Path:** `weights/lp_det.onnx`
- **Architecture:** Lightweight YOLOv8 detector
- **Input shape:** `(1, 3, 320, 320)` (vehicle crop)
- **Classes:** `0` = License Plate

### 4. `lp_ocr.onnx` (License Plate Recognition)
- **Path:** `weights/lp_ocr.onnx`
- **Architecture:** CRNN / TrOCR / PaddleOCR text recognition
- **Input shape:** `(1, 3, 48, 160)` or `(1, 3, 32, 100)`
