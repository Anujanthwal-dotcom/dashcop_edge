"""Online Cross-Association SORT (Fast CA-SORT) without Gurobi dependencies."""

import numpy as np
from typing import List, Tuple, Dict, Any
from .kalman_box_tracker import KalmanBoxTracker


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Solves the bipartite matching problem using Hungarian algorithm with pure NumPy greedy fallback."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.column_stack((row_ind, col_ind))
    except ImportError:
        # Fast greedy bipartite matching fallback
        matched_indices = []
        rows, cols = cost_matrix.shape
        used_rows = set()
        used_cols = set()

        flat_order = np.argsort(cost_matrix, axis=None)
        for idx in flat_order:
            r = idx // cols
            c = idx % cols
            if r not in used_rows and c not in used_cols:
                used_rows.add(r)
                used_cols.add(c)
                matched_indices.append((r, c))
                if len(used_rows) == rows or len(used_cols) == cols:
                    break

        if not matched_indices:
            return np.empty((0, 2), dtype=int)
        return np.array(matched_indices, dtype=int)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between two arrays of bounding boxes (xyxy format)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(1e-6, union), 0.0)


class OnlineCASORT:
    """Real-Time Online Cross-Association Multi-Class Tracker.
    
    Replaces offline research ILP/MWIS solvers (Gurobi) with low-latency
    linear sum assignment and cross-mask affinity tracking.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        assoc_iou_threshold: float = 0.25,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.assoc_iou_threshold = assoc_iou_threshold

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(
        self,
        detections: np.ndarray,
        masks: np.ndarray = None,
        cross_masks: np.ndarray = None,
    ) -> List[Dict[str, Any]]:
        """Updates tracks with current frame detections.
        
        Args:
            detections: (N, 6) [x1, y1, x2, y2, score, class_id]
                where class_id 0 = Rider, class_id 1 = Motorbike
            masks: (N, H, W) instance masks (optional)
            cross_masks: (N, H, W) cross association masks (optional)
            
        Returns:
            List of active confirmed tracks with format:
            {
                "id": int,
                "bbox": [x1, y1, x2, y2],
                "cls": int, (0=Rider, 1=Motor)
                "associated_id": int, (ID of associated motorbike if rider)
                "associated_riders": List[int] (if motor, IDs of riders on this bike)
            }
        """
        self.frame_count += 1

        # 1. Predict new locations of existing tracks
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :4] = pos
            trks[t, 4] = trk.cls_id
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        # 2. Match detections with existing trackers
        dets = np.asarray(detections) if len(detections) > 0 else np.empty((0, 6))
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)

        # 3. Update matched trackers
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(dets[d_idx, :4], cls_id=int(dets[d_idx, 5]))

        # 4. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            trk = KalmanBoxTracker(dets[d_idx, :4], cls_id=int(dets[d_idx, 5]))
            self.trackers.append(trk)

        # 5. Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        # 6. Online Cross-Association (Riders to Motorbikes)
        active_riders = [trk for trk in self.trackers if trk.cls_id == 0 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)]
        active_motors = [trk for trk in self.trackers if trk.cls_id == 1 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)]

        self._cross_associate_riders_to_motors(active_riders, active_motors, cross_masks, detections)

        # 7. Build output list
        output_tracks = []
        for trk in self.trackers:
            if (trk.hits >= self.min_hits or self.frame_count <= self.min_hits) and trk.time_since_update <= 1:
                bbox = trk.get_state().tolist()
                trk_data = {
                    "id": trk.id,
                    "bbox": bbox,
                    "cls": trk.cls_id,
                    "associated_id": trk.associated_id,
                    "associated_riders": [],
                }
                if trk.cls_id == 1:
                    trk_data["associated_riders"] = [
                        r.id for r in active_riders if r.associated_id == trk.id
                    ]
                output_tracks.append(trk_data)

        return output_tracks

    def _associate_detections_to_trackers(
        self, dets: np.ndarray, trks: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associates detections with predicted tracker boxes using class-gated IoU."""
        if len(trks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trks)))

        iou_matrix = compute_iou_matrix(dets[:, :4], trks[:, :4])

        # Enforce class gating: cost is high if classes do not match
        det_classes = dets[:, 5]
        trk_classes = trks[:, 4]
        class_mismatch = det_classes[:, None] != trk_classes[None, :]

        cost_matrix = 1.0 - iou_matrix
        cost_matrix[class_mismatch] = 10.0
        cost_matrix[iou_matrix < self.iou_threshold] = 10.0

        matches = linear_assignment(cost_matrix)

        matched_indices = []
        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(trks)))

        for d, t in matches:
            if cost_matrix[d, t] < (1.0 - self.iou_threshold):
                matched_indices.append((t, d))
                if d in unmatched_dets:
                    unmatched_dets.remove(d)
                if t in unmatched_trks:
                    unmatched_trks.remove(t)

        return matched_indices, unmatched_dets, unmatched_trks

    def _cross_associate_riders_to_motors(
        self,
        riders: List[KalmanBoxTracker],
        motors: List[KalmanBoxTracker],
        cross_masks: np.ndarray = None,
        detections: np.ndarray = None,
    ) -> None:
        """Associates each rider track to its corresponding motorcycle track."""
        if not riders or not motors:
            for r in riders:
                r.associated_id = -1
            return

        r_boxes = np.array([r.get_state() for r in riders])
        m_boxes = np.array([m.get_state() for m in motors])

        # Compute spatial containment / overlap matrix
        iou_matrix = compute_iou_matrix(r_boxes, m_boxes)

        for i, r in enumerate(riders):
            best_m_idx = -1
            best_score = -1.0
            r_center_x = (r_boxes[i, 0] + r_boxes[i, 2]) / 2.0
            r_bottom_y = r_boxes[i, 3]

            for j, m in enumerate(motors):
                m_x1, m_y1, m_x2, m_y2 = m_boxes[j]
                iou_score = iou_matrix[i, j]

                # Heuristic: rider sits on top/inside horizontal bounds of bike
                horizontal_containment = 1.0 if (m_x1 - 20 <= r_center_x <= m_x2 + 20) else 0.0
                vertical_proximity = 1.0 if (m_y1 - 30 <= r_bottom_y <= m_y2 + 50) else 0.0
                proximity_score = 0.6 * iou_score + 0.4 * (horizontal_containment * vertical_proximity)

                if proximity_score > best_score and proximity_score >= self.assoc_iou_threshold:
                    best_score = proximity_score
                    best_m_idx = j

            if best_m_idx != -1:
                r.associated_id = motors[best_m_idx].id
            else:
                r.associated_id = -1
