"""Annotated video: team ellipses, ball marker, projected pitch lines and a top-down radar."""
from pathlib import Path

import cv2
import numpy as np

from foot_cv.pipeline import TrackingResult
from foot_cv.pitch import EDGES, Pitch, PitchCanvas

REFEREE_COLOUR = (0, 220, 255)
UNKNOWN_COLOUR = (160, 160, 160)


def _colour(row, team_colours) -> tuple[int, int, int]:
    if row.cls == "referee":
        return REFEREE_COLOUR
    return team_colours.get(int(row.team), UNKNOWN_COLOUR)


def _draw_pitch_lines(frame: np.ndarray, H: np.ndarray, pitch: Pitch) -> None:
    """Re-project the pitch model into the image: if the lines sit on the real ones, H is good."""
    H_inv = np.linalg.inv(H)
    v = pitch.vertices
    for a, b in EDGES:
        seg = np.linspace(v[a - 1], v[b - 1], 20).reshape(-1, 1, 2)
        pts = cv2.perspectiveTransform(seg, H_inv).reshape(-1, 2).astype(np.int32)
        cv2.polylines(frame, [pts], False, (255, 0, 255), 2, cv2.LINE_AA)
    circle = np.array([[pitch.length / 2 + pitch.centre_circle_radius * np.cos(a),
                        pitch.width / 2 + pitch.centre_circle_radius * np.sin(a)]
                       for a in np.linspace(0, 2 * np.pi, 60)], dtype=np.float32).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(circle, H_inv).reshape(-1, 2).astype(np.int32)
    cv2.polylines(frame, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)


def render(video: Path, result: TrackingResult, out_path: Path, show_pitch_lines: bool = True) -> None:
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # H.264 so the output plays in VS Code, browsers and QuickTime.
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), result.fps, (w, h))
    canvas = PitchCanvas()
    radar_w = w // 3
    radar_h = int(radar_w * canvas.size[1] / canvas.size[0])
    by_frame = dict(tuple(result.tracking.groupby("frame")))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H = result.homographies[frame_idx] if frame_idx < len(result.homographies) else None
        if show_pitch_lines and H is not None and not np.isnan(H).any():
            _draw_pitch_lines(frame, H, canvas.pitch)

        rows = by_frame.get(frame_idx)
        radar_points, ball_xy = [], None
        if rows is not None:
            for r in rows.itertuples():
                if r.cls == "ball":
                    x, y = int(r.px), int(r.y1)
                    tri = np.array([[x, y - 4], [x - 10, y - 22], [x + 10, y - 22]], dtype=np.int32)
                    cv2.fillPoly(frame, [tri], (0, 140, 255) if r.interpolated else (0, 255, 0))
                    cv2.polylines(frame, [tri], True, (0, 0, 0), 2)
                    if not np.isnan(r.x_m):
                        ball_xy = np.array([r.x_m, r.y_m])
                    continue
                colour = _colour(r, result.team_colours)
                width = int(r.x2 - r.x1)
                cv2.ellipse(frame, (int(r.px), int(r.py)), (width, int(0.35 * width)), 0, -45, 235,
                            colour, 3, cv2.LINE_AA)
                label = f"{'GK ' if r.cls == 'goalkeeper' else ''}{r.track_id}"
                cv2.putText(frame, label, (int(r.px) - 12, int(r.py) + 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colour, 2, cv2.LINE_AA)
                if not np.isnan(r.x_m):
                    radar_points.append((np.array([r.x_m, r.y_m]), colour))

        radar = cv2.resize(canvas.draw(radar_points, ball_xy), (radar_w, radar_h))
        x0, y0 = 10, 10  # top-left, over the stands, so it never hides players
        roi = frame[y0:y0 + radar_h, x0:x0 + radar_w]
        frame[y0:y0 + radar_h, x0:x0 + radar_w] = cv2.addWeighted(roi, 0.15, radar, 0.85, 0)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
