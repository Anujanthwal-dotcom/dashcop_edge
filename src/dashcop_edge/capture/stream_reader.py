"""Low-latency streaming ingestion with dynamic Video/USB toggle and rate pacing."""

import queue
import threading
import time
from typing import Generator, Optional, Tuple, Any


class StreamReader:
    """Asynchronous thread-safe stream reader supporting rate-paced video files and live USB cameras."""

    def __init__(
        self,
        source: str = "data/sample.mp4",
        mode: str = "video",
        target_fps: int = 30,
        queue_size: int = 3,
    ):
        self.source = source
        self.mode = mode.lower()
        self.target_fps = max(1, target_fps)
        self.frame_interval = 1.0 / self.target_fps
        self.queue_size = max(1, queue_size)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_actual = float(self.target_fps)
        self._total_frames = 0
        self._dropped_frames = 0

    def start(self) -> "StreamReader":
        """Starts the capture background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stops capture and joins worker thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _capture_worker(self) -> None:
        """Background worker reading frames and enforcing real-time rate pacing."""
        try:
            import cv2
        except ImportError:
            # Synthetic camera fallback for testing without opencv installed
            self._synthetic_capture_worker()
            return

        # Determine capture source: integer device index or video file path
        cap_source: Any = self.source
        if self.mode == "usb":
            try:
                cap_source = int(self.source)
            except ValueError:
                cap_source = self.source

        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            print(f"[StreamReader] WARNING: Could not open source '{self.source}' (mode={self.mode}). Falling back to synthetic.")
            cap.release()
            self._synthetic_capture_worker()
            return

        # If video mode, try detecting native FPS
        if self.mode == "video":
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps and native_fps > 0:
                self.target_fps = int(native_fps)
                self.frame_interval = 1.0 / self.target_fps

        last_frame_time = time.time()

        while not self._stop_event.is_set():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if self.mode == "video":
                    # Loop video playback for continuous testing
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            self._total_frames += 1

            # Bounded queue logic: drop oldest frame if full to prevent buffer latency build-up
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                    self._dropped_frames += 1
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait((self._total_frames, frame, time.time()))
            except queue.Full:
                self._dropped_frames += 1

            # Real-time rate pacing for video files
            if self.mode == "video":
                elapsed = time.time() - loop_start
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

        cap.release()

    def _synthetic_capture_worker(self) -> None:
        """Generates synthetic dummy frames for testing environments where camera or opencv is absent."""
        frame_idx = 0
        while not self._stop_event.is_set():
            t_start = time.time()
            frame_idx += 1

            # Create lightweight synthetic frame structure (H, W, 3)
            try:
                import numpy as np
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            except ImportError:
                dummy_frame = [[0] * 640 for _ in range(480)]

            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                    self._dropped_frames += 1
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait((frame_idx, dummy_frame, time.time()))
            except queue.Full:
                self._dropped_frames += 1

            elapsed = time.time() - t_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def read(self, timeout: float = 0.5) -> Tuple[bool, Optional[int], Optional[Any], float]:
        """Reads the next available real-time frame."""
        try:
            frame_id, frame, timestamp = self._frame_queue.get(timeout=timeout)
            return True, frame_id, frame, timestamp
        except queue.Empty:
            return False, None, None, 0.0

    def stream(self) -> Generator[Tuple[int, Any, float], None, None]:
        """Generator yielding (frame_id, frame, timestamp) in real-time."""
        self.start()
        try:
            while not self._stop_event.is_set():
                ok, fid, frame, ts = self.read()
                if ok and frame is not None and fid is not None:
                    yield fid, frame, ts
        finally:
            self.stop()
