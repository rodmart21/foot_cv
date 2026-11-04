"""Unsupervised team assignment from shirt colours."""
import cv2
import numpy as np
from sklearn.cluster import KMeans

GRASS_LO = np.array([30, 40, 40])
GRASS_HI = np.array([90, 255, 255])
LAB_BINS = 4


def shirt_pixels(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """BGR pixels of the torso region with grass removed, shape (N, 3)."""
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = y2 - y1, x2 - x1
    crop = frame[y1 + int(0.15 * h): y1 + int(0.5 * h), x1 + int(0.2 * w): x2 - int(0.2 * w)]
    if crop.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    grass = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), GRASS_LO, GRASS_HI) > 0
    return crop[~grass]


def shirt_feature(pixels: np.ndarray) -> np.ndarray | None:
    """Normalised LAB colour histogram (LAB_BINS^3 dims)."""
    if len(pixels) < 10:
        return None
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hist, _ = np.histogramdd(lab, bins=LAB_BINS, range=[(0, 256)] * 3)
    hist = hist.ravel().astype(np.float32)
    return np.sqrt(hist / hist.sum())  # Hellinger: euclidean on sqrt-probabilities


class TeamClassifier:
    def __init__(self, n_teams: int = 2, seed: int = 0):
        self.kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=seed)

    def fit(self, features: np.ndarray) -> "TeamClassifier":
        self.kmeans.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(features)
