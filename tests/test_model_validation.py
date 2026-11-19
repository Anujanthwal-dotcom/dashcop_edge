"""Unit tests ensuring strict ONNX model presence enforcement."""

import unittest
from dashcop_edge.config import EdgeConfig
from dashcop_edge.models.onnx_engine import OnnxModelSession


class TestModelValidation(unittest.TestCase):
    def test_missing_onnx_raises_filenotfound(self):
        """Verify that attempting to load a non-existent ONNX model raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            OnnxModelSession(model_path="weights/non_existent_model.onnx")

        self.assertIn("Missing required ONNX model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
