"""Pitch geometry (metres, origin top-left corner) and top-down drawing."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Pitch:
    length: float = 105.0
    width: float = 68.0
    penalty_box_width: float = 40.32
    penalty_box_length: float = 16.5
    goal_box_width: float = 18.32
    goal_box_length: float = 5.5
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    @property
    def vertices(self) -> np.ndarray:
        """The 32 landmarks, in the order the pitch keypoint model predicts them."""
        L, W = self.length, self.width
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width, self.goal_box_length
        r, spot = self.centre_circle_radius, self.penalty_spot_distance
        return np.array([
            (0, 0), (0, (W - pbw) / 2), (0, (W - gbw) / 2), (0, (W + gbw) / 2), (0, (W + pbw) / 2), (0, W),
            (gbl, (W - gbw) / 2), (gbl, (W + gbw) / 2),
            (spot, W / 2),
            (pbl, (W - pbw) / 2), (pbl, (W - gbw) / 2), (pbl, (W + gbw) / 2), (pbl, (W + pbw) / 2),
            (L / 2, 0), (L / 2, W / 2 - r), (L / 2, W / 2 + r), (L / 2, W),
            (L - pbl, (W - pbw) / 2), (L - pbl, (W - gbw) / 2), (L - pbl, (W + gbw) / 2), (L - pbl, (W + pbw) / 2),
            (L - spot, W / 2),
            (L - gbl, (W - gbw) / 2), (L - gbl, (W + gbw) / 2),
            (L, 0), (L, (W - pbw) / 2), (L, (W - gbw) / 2), (L, (W + gbw) / 2), (L, (W + pbw) / 2), (L, W),
            (L / 2 - r, W / 2), (L / 2 + r, W / 2),
        ], dtype=np.float32)


# 1-indexed pairs of vertices joined by a pitch line.
EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
    (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
    (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
    (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
]


class PitchCanvas:
    """Renders a top-down pitch and maps metres to canvas pixels."""

    def __init__(self, pitch: Pitch = Pitch(), scale: float = 8.0, padding: float = 4.0):
        self.pitch, self.scale, self.padding = pitch, scale, padding
        self.size = (int((pitch.length + 2 * padding) * scale), int((pitch.width + 2 * padding) * scale))
        self.background = self._draw_background()

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        return ((np.asarray(xy, dtype=np.float32) + self.padding) * self.scale).astype(np.int32)

    def _draw_background(self) -> np.ndarray:
        img = np.full((self.size[1], self.size[0], 3), (60, 140, 60), dtype=np.uint8)
        v = self.to_px(self.pitch.vertices)
        line = (235, 235, 235)
        for a, b in EDGES:
            cv2.line(img, tuple(v[a - 1]), tuple(v[b - 1]), line, 2, cv2.LINE_AA)
        centre = tuple(self.to_px([self.pitch.length / 2, self.pitch.width / 2]))
        cv2.circle(img, centre, int(self.pitch.centre_circle_radius * self.scale), line, 2, cv2.LINE_AA)
        for spot in (v[8], v[21], centre):
            cv2.circle(img, tuple(spot), 4, line, -1, cv2.LINE_AA)
        return img

    def draw(self, points: list[tuple[np.ndarray, tuple[int, int, int]]], ball: np.ndarray | None = None) -> np.ndarray:
        """points: list of (xy_metres, bgr colour)."""
        img = self.background.copy()
        for xy, colour in points:
            p = tuple(self.to_px(xy))
            cv2.circle(img, p, 9, colour, -1, cv2.LINE_AA)
            cv2.circle(img, p, 9, (20, 20, 20), 2, cv2.LINE_AA)
        if ball is not None:
            p = tuple(self.to_px(ball))
            cv2.circle(img, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p, 6, (0, 0, 0), 2, cv2.LINE_AA)
        return img
