"""Segmentation and Cross-Association (SAC) Detector with ONNX Runtime."""

import numpy as np
from typing import Tuple, Optional
from .onnx_engine import OnnxModelSession


def letterbox_image(image: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resizes and pads image to target square dimension keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    import cv2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    return padded, scale, (pad_w, pad_h)


class SACDetector:
    """YOLOv8-Seg SAC (Segmentation and Cross-Association) detector.
    
    Requires trained ONNX weights (e.g. weights/sac.onnx). Raises FileNotFoundError
    if model file is missing.
    """

    def __init__(self, model_path: str = "weights/sac.onnx", provider: str = "cpu", conf_thresh: float = 0.5):
        self.conf_thresh = conf_thresh
        self.session = OnnxModelSession(model_path=model_path, provider=provider)
        self.frame_idx = 0

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Infers rider and motorcycle detections, masks, and cross-association masks.
        
        Returns:
            detections: np.ndarray (N, 6) [x1, y1, x2, y2, conf, class_id]
                where class_id 0 = Rider, class_id 1 = Motorbike
            masks: Optional[np.ndarray] (N, H, W)
            cross_masks: Optional[np.ndarray] (N, H, W)
        """
        self.frame_idx += 1

        # Standard YOLOv8 pre-processing
        padded, scale, (pad_w, pad_h) = letterbox_image(frame, 640)
        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(blob, axis=0)

        outputs = self.session.run(input_tensor)
        if not outputs or len(outputs) == 0:
            raise RuntimeError(f"[SACDetector] ONNX model '{self.session.model_path}' returned empty outputs.")

        # Parse ONNX output tensors
        pred = outputs[0]  # Expected [1, 6+num_classes, num_anchors] or [1, N, 6]
        if pred.ndim == 3 and pred.shape[1] > pred.shape[2]:
            pred = pred.transpose(0, 2, 1)

        raw_boxes = pred[0]
        valid = raw_boxes[:, 4] > self.conf_thresh
        boxes = raw_boxes[valid]

        # Rescale boxes back to original image dimensions
        orig_h, orig_w = frame.shape[:2]
        if len(boxes) > 0:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        masks = outputs[1] if len(outputs) > 1 else None
        cross_masks = outputs[2] if len(outputs) > 2 else None

        return boxes, masks, cross_masks
