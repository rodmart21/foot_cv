"""Image-to-pitch projection: from detected pitch keypoints, or from a fixed manual calibration."""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    """Fixed camera: one-parameter lens model (division model, k1 < 0 = barrel) plus a homography.

    Pixels are first undistorted (straight pitch lines become straight), then mapped to metres by H.
    """
    H: np.ndarray
    k1: float = 0.0
    size: tuple[int, int] = (1280, 960)   # image width, height

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["homography"]), d.get("k1", 0.0), tuple(d.get("image_size", (1280, 960))))

    def _norm(self) -> tuple[np.ndarray, float]:
        return np.array(self.size, dtype=np.float64) / 2, float(self.size[0])

    def undistort(self, px: np.ndarray) -> np.ndarray:
        c, f = self._norm()
        u = (np.asarray(px, dtype=np.float64).reshape(-1, 2) - c) / f
        return c + f * u / (1 + self.k1 * (u ** 2).sum(1, keepdims=True))

    def distort(self, px: np.ndarray) -> np.ndarray:
        """Inverse of undistort, in closed form: r_v = r_u / (1 + k r_u^2) solved for r_u.
        Points with no valid solution (outside the lens model's range) come back as NaN."""
        c, f = self._norm()
        v = (np.asarray(px, dtype=np.float64).reshape(-1, 2) - c) / f
        if self.k1 == 0:
            return c + f * v
        rv = np.hypot(v[:, 0], v[:, 1])[:, None]
        disc = 1 - 4 * self.k1 * rv ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            ru = np.where(rv > 1e-12, (1 - np.sqrt(disc)) / (2 * self.k1 * rv), 0.0)
            u = np.where(rv > 1e-12, v * ru / rv, v)
        u[(disc[:, 0] < 0) | (ru[:, 0] < 0)] = np.nan
        return c + f * u

    def to_pitch(self, px: np.ndarray) -> np.ndarray:
        pts = self.undistort(px).astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pm: np.ndarray) -> np.ndarray:
        pts = np.asarray(pm, dtype=np.float32).reshape(-1, 1, 2)
        return self.distort(cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2))


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
