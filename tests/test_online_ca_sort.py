"""Unit tests for Fast Online CA-SORT Tracker."""

import unittest
import numpy as np
from dashcop_edge.tracker.online_ca_sort import OnlineCASORT


class TestOnlineCASORT(unittest.TestCase):
    def test_multi_class_tracking_and_cross_association(self):
        tracker = OnlineCASORT(max_age=10, min_hits=1, iou_threshold=0.2)

        # Bike at [200, 200, 300, 350], Rider 1 at [220, 150, 260, 240], Rider 2 at [250, 150, 290, 240]
        bike = [200, 200, 300, 350, 0.9, 1]  # cls 1 = motor
        rider1 = [220, 150, 260, 240, 0.85, 0]  # cls 0 = rider
        rider2 = [250, 150, 290, 240, 0.82, 0]  # cls 0 = rider

        dets = np.array([bike, rider1, rider2], dtype=np.float32)

        tracks = tracker.update(dets)
        self.assertEqual(len(tracks), 3)

        motors = [t for t in tracks if t["cls"] == 1]
        riders = [t for t in tracks if t["cls"] == 0]

        self.assertEqual(len(motors), 1)
        self.assertEqual(len(riders), 2)

        # Verify cross association
        motor_id = motors[0]["id"]
        for r in riders:
            self.assertEqual(r["associated_id"], motor_id)

        self.assertIn(riders[0]["id"], motors[0]["associated_riders"])
        self.assertIn(riders[1]["id"], motors[0]["associated_riders"])


if __name__ == "__main__":
    unittest.main()
