"""Stateful temporal persistence filter for traffic violation detection."""

from collections import deque
from typing import Dict, List, Any, Optional
import numpy as np
from .best_shot import BestShotSelector


class ViolationStateManager:
    """Manages temporal voting and single-ticket lifecycle per vehicle track."""

    def __init__(self, min_consecutive_frames: int = 5, buffer_size: int = 15):
        self.min_consecutive_frames = min_consecutive_frames
        self.best_shot_selector = BestShotSelector(max_buffer_size=buffer_size)

        # track_id -> deque of bools (is_triple_riding)
        self._triple_history: Dict[int, deque] = {}
        # track_id -> deque of bools (has_helmet_violation)
        self._helmet_history: Dict[int, deque] = {}
        # track_id -> set of emitted violation types (prevents duplicate tickets)
        self._emitted_violations: Dict[int, set] = {}

    def update_track(
        self,
        frame_id: int,
        frame: np.ndarray,
        motor_track: Dict[str, Any],
        rider_tracks: List[Dict[str, Any]],
        hnh_results: Dict[int, bool],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """Updates violation state for a motorcycle track across time.
        
        Returns:
            List of newly confirmed violation events ready for ticket dispatch.
        """
        motor_id = motor_track["id"]
        bbox = motor_track["bbox"]
        associated_rider_ids = motor_track.get("associated_riders", [])
        num_riders = len(associated_rider_ids)

        if motor_id not in self._triple_history:
            self._triple_history[motor_id] = deque(maxlen=self.min_consecutive_frames * 2)
            self._helmet_history[motor_id] = deque(maxlen=self.min_consecutive_frames * 2)
            self._emitted_violations[motor_id] = set()

        # Check frame-level conditions
        is_triple = (num_riders >= 3)
        has_helmet_violation = False
        for r_id in associated_rider_ids:
            if hnh_results.get(r_id, False):
                has_helmet_violation = True
                break

        self._triple_history[motor_id].append(is_triple)
        self._helmet_history[motor_id].append(has_helmet_violation)

        # Store vehicle crop in best-shot ring buffer
        h, w = frame.shape[:2] if hasattr(frame, "shape") else (480, 640)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 20
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(w, x2 + pad)
        crop_y2 = min(h, y2 + pad)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2] if hasattr(frame, "__getitem__") else np.zeros((10, 10))

        self.best_shot_selector.add_candidate(motor_id, frame_id, crop, bbox, timestamp)

        # Evaluate temporal persistence
        confirmed_events = []
        triple_count = sum(self._triple_history[motor_id])
        helmet_count = sum(self._helmet_history[motor_id])

        # 1. Triple Riding Trigger
        if triple_count >= self.min_consecutive_frames and "TRIPLE_RIDING" not in self._emitted_violations[motor_id]:
            self._emitted_violations[motor_id].add("TRIPLE_RIDING")
            best = self.best_shot_selector.select_best(motor_id)
            if best:
                confirmed_events.append({
                    "track_id": motor_id,
                    "violation_type": "TRIPLE_RIDING",
                    "num_riders": num_riders,
                    "frame_id": best["frame_id"],
                    "crop": best["crop"],
                    "bbox": best["bbox"],
                    "timestamp": best["timestamp"],
                })

        # 2. Helmet Violation Trigger
        if helmet_count >= self.min_consecutive_frames and "NO_HELMET" not in self._emitted_violations[motor_id]:
            self._emitted_violations[motor_id].add("NO_HELMET")
            best = self.best_shot_selector.select_best(motor_id)
            if best:
                confirmed_events.append({
                    "track_id": motor_id,
                    "violation_type": "NO_HELMET",
                    "num_riders": num_riders,
                    "frame_id": best["frame_id"],
                    "crop": best["crop"],
                    "bbox": best["bbox"],
                    "timestamp": best["timestamp"],
                })

        return confirmed_events

    def cleanup_dead_tracks(self, active_track_ids: List[int]) -> None:
        """Removes buffers for tracks that have left the scene."""
        current_tracked = set(active_track_ids)
        dead = [tid for tid in self._triple_history if tid not in current_tracked]
        for tid in dead:
            self._triple_history.pop(tid, None)
            self._helmet_history.pop(tid, None)
            self._emitted_violations.pop(tid, None)
            self.best_shot_selector.clear(tid)
