"""Kalman Filter based bounding box tracker for online tracking."""

import numpy as np


class KalmanBoxTracker:
    """Tracks an individual bounding box using a constant velocity Kalman Filter model."""

    count = 0

    def __init__(self, bbox, cls_id: int = 0):
        """Initializes tracker with initial bounding box [x1, y1, x2, y2] and class id."""
        # State vector: [x_center, y_center, area, aspect_ratio, vx, vy, va]
        self.cls_id = int(cls_id)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.time_since_update = 0
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.associated_id = -1  # If rider, points to motor ID; if motor, ID is self

        # Initialize state mean [7x1]
        self.x = np.zeros((7, 1), dtype=np.float32)
        self.x[:4] = self._bbox_to_z(bbox)

        # State transition matrix F [7x7]
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 4] = 1.0  # x = x + vx
        self.F[1, 5] = 1.0  # y = y + vy
        self.F[2, 6] = 1.0  # s = s + vs

        # Measurement matrix H [4x7]
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)

        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]).astype(np.float32)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)

    @staticmethod
    def _bbox_to_z(bbox):
        """Converts [x1, y1, x2, y2] into [x_center, y_center, area, aspect_ratio]."""
        w = max(1e-4, float(bbox[2] - bbox[0]))
        h = max(1e-4, float(bbox[3] - bbox[1]))
        x = float(bbox[0] + w / 2.0)
        y = float(bbox[1] + h / 2.0)
        s = float(w * h)
        r = float(w / h)
        return np.array([[x], [y], [s], [r]], dtype=np.float32)

    @staticmethod
    def _x_to_bbox(x):
        """Converts [x_center, y_center, area, aspect_ratio] to [x1, y1, x2, y2]."""
        s = max(1e-4, float(x[2, 0]))
        r = max(1e-4, float(x[3, 0]))
        w = np.sqrt(s * r)
        h = s / max(1e-4, w)
        x1 = float(x[0, 0] - w / 2.0)
        y1 = float(x[1, 0] - h / 2.0)
        x2 = float(x[0, 0] + w / 2.0)
        y2 = float(x[1, 0] + h / 2.0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self):
        """Advances state vector using transition matrix F."""
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._x_to_bbox(self.x))
        return self.history[-1]

    def update(self, bbox, cls_id: int = 0):
        """Updates the state vector with observed bounding box."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.cls_id = int(cls_id)

        z = self._bbox_to_z(bbox)
        # Innovation y = z - Hx
        y = z - np.dot(self.H, self.x)
        # Innovation covariance S = H P H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Kalman gain K = P H^T S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Updated state and covariance
        self.x = self.x + np.dot(K, y)
        I = np.eye(7, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self):
        """Returns the current bounding box estimate [x1, y1, x2, y2]."""
        return self._x_to_bbox(self.x)
