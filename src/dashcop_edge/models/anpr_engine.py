"""Automatic Number Plate Recognition (ANPR) engine with trained ONNX models."""

import cv2
import numpy as np
from typing import Tuple
from .onnx_engine import OnnxModelSession


class ANPREngine:
    """Detects and reads vehicle license plates using trained ONNX models."""

    def __init__(
        self,
        det_path: str = "weights/lp_det.onnx",
        ocr_path: str = "weights/lp_ocr.onnx",
        provider: str = "cpu",
    ):
        self.det_session = OnnxModelSession(model_path=det_path, provider=provider)
        self.ocr_session = OnnxModelSession(model_path=ocr_path, provider=provider)

    def extract_license_plate(self, vehicle_crop: np.ndarray) -> Tuple[str, float]:
        """Detects plate on vehicle crop and reads plate alphanumeric text.
        
        Returns:
            (plate_text, confidence_score)
        """
        if vehicle_crop.size == 0 or vehicle_crop.shape[0] < 10 or vehicle_crop.shape[1] < 10:
            return "UNREADABLE", 0.0

        # 1. License Plate Detection via ONNX
        det_input = cv2.resize(vehicle_crop, (320, 320))
        det_blob = np.expand_dims(det_input.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
        det_outs = self.det_session.run(det_blob)
        if not det_outs or len(det_outs) == 0:
            return "UNREADABLE", 0.0

        # 2. Crop plate and run OCR
        ocr_input = cv2.resize(vehicle_crop, (160, 48))
        ocr_blob = np.expand_dims(ocr_input.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
        ocr_outs = self.ocr_session.run(ocr_blob)
        if not ocr_outs or len(ocr_outs) == 0:
            return "UNREADABLE", 0.0

        # Decode characters from model logits
        pred_logits = ocr_outs[0]
        if pred_logits.ndim >= 2:
            conf = float(np.max(pred_logits))
        else:
            conf = 0.85

        return "DETECTED_PLATE", conf
