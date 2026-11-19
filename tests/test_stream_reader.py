"""Unit tests for asynchronous stream reader."""

import time
import unittest
from dashcop_edge.capture.stream_reader import StreamReader


class TestStreamReader(unittest.TestCase):
    def test_synthetic_stream_reader(self):
        reader = StreamReader(source="dummy.mp4", mode="video", target_fps=60, queue_size=5)
        reader.start()

        frames_received = 0
        for _ in range(5):
            ok, fid, frame, ts = reader.read(timeout=0.5)
            if ok:
                frames_received += 1
                self.assertIsNotNone(frame)
                self.assertGreater(ts, 0.0)

        reader.stop()
        self.assertGreater(frames_received, 0)


if __name__ == "__main__":
    unittest.main()
