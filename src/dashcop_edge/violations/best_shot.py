"""Best-Shot image quality assessment for vehicle crops."""

import numpy as np
from typing import List, Dict, Any, Optional


def compute_sharpness_score(image: np.ndarray) -> float:
    """Computes sharpness using the variance of the Laplacian filter."""
    if image is None or image.size == 0 or image.shape[0] < 5 or image.shape[1] < 5:
        return 0.0

    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        score = float(laplacian.var())
        return score
    except Exception:
        # Fallback variance of raw gradient
        return float(np.var(image))


class BestShotSelector:
    """Maintains a bounded circular buffer of candidate frames per track and selects the sharpest crop."""

    def __init__(self, max_buffer_size: int = 15):
        self.max_buffer_size = max_buffer_size
        self._buffers: Dict[int, List[Dict[str, Any]]] = {}

    def add_candidate(self, track_id: int, frame_id: int, crop: np.ndarray, bbox: List[float], timestamp: float) -> None:
        """Adds a candidate vehicle crop to track buffer."""
        if track_id not in self._buffers:
            self._buffers[track_id] = []

        buf = self._buffers[track_id]
        sharpness = compute_sharpness_score(crop)
        h, w = crop.shape[:2] if hasattr(crop, "shape") else (0, 0)
        area = float(w * h)
        # Quality metric balances edge sharpness and resolution
        quality_score = sharpness * np.sqrt(max(1.0, area))

        buf.append({
            "frame_id": frame_id,
            "crop": crop,
            "bbox": bbox,
            "timestamp": timestamp,
            "quality": quality_score,
            "width": w,
            "height": h,
        })

        if len(buf) > self.max_buffer_size:
            buf.pop(0)

    def select_best(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Returns the optimal candidate crop for license plate extraction."""
        buf = self._buffers.get(track_id, [])
        if not buf:
            return None

        # Return candidate with maximum quality score
        best = max(buf, key=lambda item: item["quality"])
        return best

    def clear(self, track_id: int) -> None:
        """Cleans up buffer for terminated tracks."""
        self._buffers.pop(track_id, None)
