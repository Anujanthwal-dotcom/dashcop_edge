"""Unit tests for configuration loading."""

import os
import unittest
from dashcop_edge.config import EdgeConfig


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = EdgeConfig()
        self.assertEqual(cfg.source.mode, "video")
        self.assertEqual(cfg.source.target_fps, 30)
        self.assertEqual(cfg.violations.min_consecutive_frames, 5)
        self.assertEqual(cfg.models.provider, "cpu")

    def test_load_yaml(self):
        yaml_path = "configs/edge_config.yaml"
        if os.path.exists(yaml_path):
            cfg = EdgeConfig.load_from_yaml(yaml_path)
            self.assertIn(cfg.source.mode, ["video", "usb"])
            self.assertGreater(cfg.source.target_fps, 0)


if __name__ == "__main__":
    unittest.main()
