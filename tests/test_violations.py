"""Unit tests for violation state management and ticket generation."""

import unittest
import numpy as np
from dashcop_edge.violations.state_manager import ViolationStateManager
from dashcop_edge.violations.ticket_dispatcher import TicketDispatcher
from dashcop_edge.models.anpr_engine import ANPREngine


class TestViolations(unittest.TestCase):
    def test_triple_riding_persistence(self):
        state_mgr = ViolationStateManager(min_consecutive_frames=3, buffer_size=5)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        motor_track = {
            "id": 10,
            "bbox": [100.0, 100.0, 200.0, 200.0],
            "cls": 1,
            "associated_riders": [1, 2, 3],  # 3 riders -> Triple riding!
        }

        # Frame 1: not confirmed yet
        events = state_mgr.update_track(1, dummy_frame, motor_track, [], {}, 100.0)
        self.assertEqual(len(events), 0)

        # Frame 2: not confirmed yet
        events = state_mgr.update_track(2, dummy_frame, motor_track, [], {}, 100.1)
        self.assertEqual(len(events), 0)

        # Frame 3: confirmed!
        events = state_mgr.update_track(3, dummy_frame, motor_track, [], {}, 100.2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["violation_type"], "TRIPLE_RIDING")
        self.assertEqual(events[0]["track_id"], 10)

        # Frame 4: should NOT emit duplicate ticket for same track
        events = state_mgr.update_track(4, dummy_frame, motor_track, [], {}, 100.3)
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()
