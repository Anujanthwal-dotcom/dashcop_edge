"""Unified ONNX Runtime inference session with strict model existence enforcement."""

import os
import sys
from typing import Optional, List, Dict, Any
import numpy as np


class OnnxModelSession:
    """Manages an ONNX Runtime InferenceSession. Throws an error and stops if model is missing."""

    def __init__(self, model_path: str, provider: str = "cpu"):
        self.model_path = model_path
        self.provider = provider.lower()
        self.session = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []

        self._initialize_session()

    def _initialize_session(self) -> None:
        """Loads the ONNX model using onnxruntime. Raises FileNotFoundError if missing."""
        if not os.path.isfile(self.model_path):
            abs_path = os.path.abspath(self.model_path)
            error_message = (
                f"\n[ONNX Engine] FATAL ERROR: Required model file not found!\n"
                f"  Missing file : '{self.model_path}'\n"
                f"  Absolute path: '{abs_path}'\n"
                f"  Execution stopped. Please place the trained ONNX model at this path.\n"
            )
            print(error_message, file=sys.stderr)
            raise FileNotFoundError(f"Missing required ONNX model: '{self.model_path}'")

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "[ONNX Engine] 'onnxruntime' package is not installed. "
                "Please run 'uv sync' or install onnxruntime to run the system."
            ) from e

        # Select execution providers
        available_providers = ort.get_available_providers()
        providers_to_use = ["CPUExecutionProvider"]

        if self.provider == "tensorrt" and "TensorrtExecutionProvider" in available_providers:
            providers_to_use = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.provider in ("cuda", "gpu") and "CUDAExecutionProvider" in available_providers:
            providers_to_use = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers_to_use)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            print(f"[ONNX Engine] Successfully loaded '{self.model_path}' using providers: {self.session.get_providers()}")
        except Exception as e:
            raise RuntimeError(
                f"[ONNX Engine] Failed to initialize ONNX session for '{self.model_path}': {e}"
            ) from e

    def run(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Runs model inference. Raises RuntimeError if execution fails."""
        if self.session is None or self.input_name is None:
            raise RuntimeError(f"[ONNX Engine] Session for '{self.model_path}' is not initialized.")

        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            return outputs
        except Exception as e:
            raise RuntimeError(f"[ONNX Engine] Inference failed for '{self.model_path}': {e}") from e
