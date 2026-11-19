"""Real-Time HUD visualizer for live edge monitoring."""

from typing import List, Dict, Any, Tuple
import numpy as np


class HUDVisualizer:
    """Renders informative live overlays on video frames."""

    def __init__(self, window_name: str = "DashCop Edge Real-Time Monitor"):
        self.window_name = window_name

    def render(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        hnh_results: Dict[int, bool],
        fps: float,
        latency_ms: float,
        frame_id: int,
        mode: str = "video",
    ) -> np.ndarray:
        """Renders bounding boxes, association links, and telemetry HUD on frame."""
        if frame is None or not hasattr(frame, "shape"):
            return frame

        try:
            import cv2
        except ImportError:
            return frame

        canvas = frame.copy()

        # Separate riders and motors
        riders = [t for t in tracks if t["cls"] == 0]
        motors = [t for t in tracks if t["cls"] == 1]
        motor_map = {m["id"]: m for m in motors}

        # 1. Draw Association Links between Riders and their Motorbikes
        for r in riders:
            assoc_id = r.get("associated_id", -1)
            if assoc_id != -1 and assoc_id in motor_map:
                m = motor_map[assoc_id]
                rx = int((r["bbox"][0] + r["bbox"][2]) / 2.0)
                ry = int((r["bbox"][1] + r["bbox"][3]) / 2.0)
                mx = int((m["bbox"][0] + m["bbox"][2]) / 2.0)
                my = int((m["bbox"][1] + m["bbox"][3]) / 2.0)
                # Dotted yellow association line
                cv2.line(canvas, (rx, ry), (mx, my), (0, 230, 255), 2, cv2.LINE_AA)

        # 2. Draw Motorbike Bounding Boxes
        has_active_violation = False
        for m in motors:
            x1, y1, x2, y2 = [int(v) for v in m["bbox"]]
            riders_on_bike = m.get("associated_riders", [])
            is_triple = len(riders_on_bike) >= 3
            has_no_helmet = any(hnh_results.get(rid, False) for rid in riders_on_bike)

            is_violating = is_triple or has_no_helmet
            if is_violating:
                has_active_violation = True
                box_color = (0, 0, 255)  # Red for violation
                thickness = 3
            else:
                box_color = (0, 255, 0)  # Green for compliant
                thickness = 2

            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

            label = f"Bike #{m['id']} | Riders: {len(riders_on_bike)}"
            if is_triple:
                label += " [TRIPLE!]"
            if has_no_helmet:
                label += " [NO HELMET!]"

            # Label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), box_color, -1)
            cv2.putText(canvas, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Rider Bounding Boxes
        for r in riders:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            r_no_helmet = hnh_results.get(r["id"], False)
            r_color = (0, 140, 255) if r_no_helmet else (255, 200, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), r_color, 1)
            r_label = f"Rider #{r['id']}" + (" (No Helmet)" if r_no_helmet else "")
            cv2.putText(canvas, r_label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, r_color, 1, cv2.LINE_AA)

        # 4. Top Telemetry Dashboard
        h, w = canvas.shape[:2]
        hud_bg_h = 42
        cv2.rectangle(canvas, (0, 0), (w, hud_bg_h), (25, 25, 25), -1)

        status_text = f"FPS: {fps:4.1f} | Latency: {latency_ms:4.1f}ms | Mode: {mode.upper()} | Frame: {frame_id} | Tracks: {len(tracks)}"
        cv2.putText(canvas, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        # 5. Flashing Violation Alert Banner
        if has_active_violation:
            banner_text = ">>> VIOLATION DETECTED <<<"
            (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            cv2.rectangle(canvas, (w - bw - 30, 6), (w - 10, 36), (0, 0, 220), -1)
            cv2.putText(canvas, banner_text, (w - bw - 20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas
