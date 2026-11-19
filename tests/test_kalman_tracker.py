"""Unit tests for Kalman Box Tracker."""

import unittest
import numpy as np
from dashcop_edge.tracker.kalman_box_tracker import KalmanBoxTracker


class TestKalmanTracker(unittest.TestCase):
    def test_tracker_lifecycle(self):
        initial_box = np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)
        tracker = KalmanBoxTracker(initial_box, cls_id=1)

        self.assertEqual(tracker.cls_id, 1)
        self.assertEqual(tracker.hits, 1)

        # Predict next position
        pred_box = tracker.predict()
        self.assertEqual(len(pred_box), 4)

        # Update with new measurement
        meas_box = np.array([105.0, 105.0, 205.0, 205.0], dtype=np.float32)
        tracker.update(meas_box, cls_id=1)
        self.assertEqual(tracker.hits, 2)
        self.assertEqual(tracker.time_since_update, 0)

        # State check
        state = tracker.get_state()
        self.assertAlmostEqual(state[0], 105.0, delta=15.0)


if __name__ == "__main__":
    unittest.main()
