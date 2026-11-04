"""Image-to-pitch projection from detected pitch keypoints."""
import cv2
import numpy as np


class ViewTransformer:
    """Holds the latest valid image->pitch homography and projects points with it."""

    def __init__(self, pitch_vertices: np.ndarray, min_points: int = 4, kp_conf: float = 0.5):
        self.pitch_vertices = pitch_vertices
        self.min_points = min_points
        self.kp_conf = kp_conf
        self.H: np.ndarray | None = None

    def update(self, keypoints_xy: np.ndarray, keypoints_conf: np.ndarray) -> bool:
        """Refit H from one frame's 32 keypoints. Keeps the previous H if the fit fails."""
        mask = keypoints_conf > self.kp_conf
        if mask.sum() < self.min_points:
            return False
        H, _ = cv2.findHomography(keypoints_xy[mask], self.pitch_vertices[mask], cv2.RANSAC, 1.0)
        if H is None:
            return False
        self.H = H
        return True

    def transform(self, points_px: np.ndarray) -> np.ndarray:
        if self.H is None or len(points_px) == 0:
            return np.full((len(points_px), 2), np.nan, dtype=np.float32)
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
