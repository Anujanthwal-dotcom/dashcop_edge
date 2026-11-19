"""Runtime configuration module for DashCop Edge."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceConfig:
    mode: str = "video"  # "video" or "usb"
    path: str = "data/sample.mp4"
    target_fps: int = 30
    queue_size: int = 3


@dataclass
class ModelsConfig:
    weights_dir: str = "weights"
    sac_model: str = "weights/sac.onnx"
    hnh_model: str = "weights/hnh.onnx"
    lp_det_model: str = "weights/lp_det.onnx"
    lp_ocr_model: str = "weights/lp_ocr.onnx"
    provider: str = "cpu"  # "cpu", "cuda", or "tensorrt"
    confidence_threshold: float = 0.5


@dataclass
class TrackerConfig:
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    assoc_iou_threshold: float = 0.25


@dataclass
class ViolationsConfig:
    min_consecutive_frames: int = 5
    best_shot_buffer_size: int = 15
    output_dir: str = "output/violations"


@dataclass
class DisplayConfig:
    enabled: bool = True
    window_name: str = "DashCop Edge Real-Time Monitor"


@dataclass
class EdgeConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    violations: ViolationsConfig = field(default_factory=ViolationsConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "configs/edge_config.yaml") -> "EdgeConfig":
        """Loads configuration from YAML file, falling back to defaults if missing or PyYAML uninstalled."""
        cfg = cls()
        if not os.path.exists(yaml_path):
            return cfg

        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            if "source" in data:
                s = data["source"]
                cfg.source = SourceConfig(
                    mode=s.get("mode", cfg.source.mode),
                    path=str(s.get("path", cfg.source.path)),
                    target_fps=int(s.get("target_fps", cfg.source.target_fps)),
                    queue_size=int(s.get("queue_size", cfg.source.queue_size)),
                )
            if "models" in data:
                m = data["models"]
                cfg.models = ModelsConfig(
                    weights_dir=m.get("weights_dir", cfg.models.weights_dir),
                    sac_model=m.get("sac_model", cfg.models.sac_model),
                    hnh_model=m.get("hnh_model", cfg.models.hnh_model),
                    lp_det_model=m.get("lp_det_model", cfg.models.lp_det_model),
                    lp_ocr_model=m.get("lp_ocr_model", cfg.models.lp_ocr_model),
                    provider=m.get("provider", cfg.models.provider),
                    confidence_threshold=float(m.get("confidence_threshold", cfg.models.confidence_threshold)),
                )
            if "tracker" in data:
                t = data["tracker"]
                cfg.tracker = TrackerConfig(
                    max_age=int(t.get("max_age", cfg.tracker.max_age)),
                    min_hits=int(t.get("min_hits", cfg.tracker.min_hits)),
                    iou_threshold=float(t.get("iou_threshold", cfg.tracker.iou_threshold)),
                    assoc_iou_threshold=float(t.get("assoc_iou_threshold", cfg.tracker.assoc_iou_threshold)),
                )
            if "violations" in data:
                v = data["violations"]
                cfg.violations = ViolationsConfig(
                    min_consecutive_frames=int(v.get("min_consecutive_frames", cfg.violations.min_consecutive_frames)),
                    best_shot_buffer_size=int(v.get("best_shot_buffer_size", cfg.violations.best_shot_buffer_size)),
                    output_dir=v.get("output_dir", cfg.violations.output_dir),
                )
            if "display" in data:
                d = data["display"]
                cfg.display = DisplayConfig(
                    enabled=bool(d.get("enabled", cfg.display.enabled)),
                    window_name=d.get("window_name", cfg.display.window_name),
                )
        except Exception:
            pass  # Fall back to default config if yaml cannot be parsed
        return cfg
