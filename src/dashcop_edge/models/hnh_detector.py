"""Helmet / No-Helmet (HNH) compliance detector."""

import cv2
import numpy as np
from typing import List, Dict, Any
from .onnx_engine import OnnxModelSession


class HNHDetector:
    """Evaluates helmet compliance on rider bounding boxes using trained ONNX model."""

    def __init__(self, model_path: str = "weights/hnh.onnx", provider: str = "cpu", conf_thresh: float = 0.5):
        self.conf_thresh = conf_thresh
        self.session = OnnxModelSession(model_path=model_path, provider=provider)

    def evaluate_riders(self, frame: np.ndarray, rider_tracks: List[Dict[str, Any]]) -> Dict[int, bool]:
        """Evaluates helmet compliance for each active rider.
        
        Returns:
            Dict mapping rider_track_id -> has_violation (True if NO-HELMET, False if compliant)
        """
        results: Dict[int, bool] = {}
        if not rider_tracks:
            return results

        orig_h, orig_w = frame.shape[:2]
        for r in rider_tracks:
            bbox = r["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Extract upper 35% of rider bounding box (head region)
            head_h = max(10, int((y2 - y1) * 0.35))
            head_y2 = min(orig_h, y1 + head_h)
            head_crop = frame[max(0, y1) : head_y2, max(0, x1) : min(orig_w, x2)]

            if head_crop.size == 0 or head_crop.shape[0] < 5 or head_crop.shape[1] < 5:
                results[r["id"]] = False
                continue

            crop_resized = cv2.resize(head_crop, (128, 128))
            blob = np.expand_dims(crop_resized.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
            outputs = self.session.run(blob)
            if outputs and len(outputs) > 0:
                pred_class = int(np.argmax(outputs[0]))
                # Class 0: No-Helmet (violation), Class 1: Helmet
                results[r["id"]] = (pred_class == 0)
            else:
                results[r["id"]] = False

        return results
