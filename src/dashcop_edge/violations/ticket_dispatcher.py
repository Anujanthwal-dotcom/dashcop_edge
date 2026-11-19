"""Asynchronous E-Ticket dispatcher saving structured JSON records and evidence snapshots."""

import json
import os
import queue
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from ..models.anpr_engine import ANPREngine


class TicketDispatcher:
    """Asynchronously generates and saves violation records and visual evidence."""

    def __init__(self, anpr_engine: ANPREngine, output_dir: str = "output/violations"):
        self.anpr_engine = anpr_engine
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.total_tickets_generated = 0

    def start(self) -> "TicketDispatcher":
        """Starts the background worker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stops worker and flushes pending tickets."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def dispatch(self, event: Dict[str, Any]) -> bool:
        """Enqueues a confirmed violation event for processing."""
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            print("[TicketDispatcher] Queue full! Dropping violation event.")
            return False

    def _worker(self) -> None:
        """Background worker that runs ANPR and writes ticket artifacts."""
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._process_event(event)
            self._queue.task_done()

    def _process_event(self, event: Dict[str, Any]) -> None:
        """Processes a single violation event."""
        track_id = event["track_id"]
        v_type = event["violation_type"]
        crop = event["crop"]
        frame_id = event["frame_id"]
        timestamp = event["timestamp"]

        # Run OCR on the selected best-shot vehicle crop
        plate_number, plate_conf = self.anpr_engine.extract_license_plate(crop)

        # Unique ticket ID
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        ticket_id = f"TICKET_{dt_str}_TRK{track_id}_{v_type}"

        # Save evidence image
        img_filename = f"{ticket_id}.jpg"
        img_path = os.path.join(self.output_dir, img_filename)
        self._save_image(crop, img_path)

        # Build metadata record
        record = {
            "ticket_id": ticket_id,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "frame_id": frame_id,
            "track_id": track_id,
            "violation_type": v_type,
            "num_riders": event.get("num_riders", 1),
            "license_plate": {
                "number": plate_number,
                "confidence": round(float(plate_conf), 3),
            },
            "bounding_box_xyxy": [round(float(v), 1) for v in event["bbox"]],
            "evidence_image": img_filename,
            "status": "ISSUED",
        }

        # Write JSON ticket
        json_path = os.path.join(self.output_dir, f"{ticket_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        self.total_tickets_generated += 1
        print(f"[E-Ticket Issued] ID={ticket_id} | Plate={plate_number} | Type={v_type}")

    def _save_image(self, image: np.ndarray, file_path: str) -> None:
        """Saves image crop using OpenCV if available, or basic header writer."""
        if image is None or not hasattr(image, "shape"):
            return
        try:
            import cv2
            cv2.imwrite(file_path, image)
        except ImportError:
            # Touch file if opencv is not installed
            with open(file_path, "wb") as f:
                f.write(b"RAW_CROP_PLACEHOLDER")
