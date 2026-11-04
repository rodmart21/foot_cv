"""Manual camera calibration: click known pitch landmarks on one frame, get a pixel->metre homography.

For fixed cameras (Alfheim, tactical cams) one calibration holds for the whole match.

    python scripts/calibrate_camera.py data/raw/alfheim/2013-11-03_first_1_600s_9s.mp4

Controls (in the image window):
    click   place the requested landmark        s   skip it (not visible)
    u       undo the last point                 Enter   finish (needs >= 4 points)
    Esc     quit without saving
Then, on the result screen:  s = save, r = redo, Esc = quit.

Coordinates: x along the pitch from the LEFT goal line (as seen in the image) to the right,
y across from the FAR touchline (top of the image) to the near one.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.homography import Calibration
from src.pitch import Pitch, PitchCanvas
from src.render import _draw_pitch_lines

WINDOW = "calibration"
MAX_DISPLAY_WIDTH = 1500
ZOOM, ZOOM_HALF = 4, 40   # magnifier: 4x of an 80x80 px area around the cursor


def landmarks(p: Pitch) -> list[tuple[str, tuple[float, float]]]:
    """Painted, clickable pitch points (line intersections and spots), in metres."""
    L, W = p.length, p.width
    pb_far, pb_near = (W - p.penalty_box_width) / 2, (W + p.penalty_box_width) / 2
    gb_far, gb_near = (W - p.goal_box_width) / 2, (W + p.goal_box_width) / 2
    pts = [
        ("halfway line x far touchline", (L / 2, 0)),
        ("centre circle x halfway line (far)", (L / 2, W / 2 - p.centre_circle_radius)),
        ("centre spot", (L / 2, W / 2)),
        ("centre circle x halfway line (near)", (L / 2, W / 2 + p.centre_circle_radius)),
        ("halfway line x near touchline", (L / 2, W)),
        ("centre circle, leftmost point", (L / 2 - p.centre_circle_radius, W / 2)),
        ("centre circle, rightmost point", (L / 2 + p.centre_circle_radius, W / 2)),
    ]
    for side, x0, sign in (("left", 0.0, 1), ("right", L, -1)):
        pts += [
            (f"{side} corner (far)", (x0, 0)),
            (f"{side} penalty box on goal line (far)", (x0, pb_far)),
            (f"{side} goal box on goal line (far)", (x0, gb_far)),
            (f"{side} goal box on goal line (near)", (x0, gb_near)),
            (f"{side} penalty box on goal line (near)", (x0, pb_near)),
            (f"{side} corner (near)", (x0, W)),
            (f"{side} goal box front corner (far)", (x0 + sign * p.goal_box_length, gb_far)),
            (f"{side} goal box front corner (near)", (x0 + sign * p.goal_box_length, gb_near)),
            (f"{side} penalty spot", (x0 + sign * p.penalty_spot_distance, W / 2)),
            (f"{side} penalty box front corner (far)", (x0 + sign * p.penalty_box_length, pb_far)),
            (f"{side} penalty box front corner (near)", (x0 + sign * p.penalty_box_length, pb_near)),
        ]
    return pts


def fit(points: list[dict], k1: float = 0.0, size: tuple[int, int] = (1280, 960)
        ) -> tuple[Calibration | None, list[float]]:
    """Calibration (lens k1 + homography) from clicked points, plus each point's error (m).
    Returns (None, inf errors) when the points cannot define a homography (e.g. all on one line)."""
    lens = Calibration(np.eye(3), k1, size)
    px = lens.undistort([q["pixel"] for q in points]).astype(np.float32)
    pm = np.array([q["pitch"] for q in points], dtype=np.float32)
    H, _ = cv2.findHomography(px, pm, 0)  # plain least squares: every clicked point counts
    if H is None:
        return None, [float("inf")] * len(points)
    calib = Calibration(H, k1, size)
    proj = calib.to_pitch([q["pixel"] for q in points])
    return calib, np.hypot(*(proj - pm).T).round(2).tolist()


def leave_one_out(points: list[dict], k1: float = 0.0, size: tuple[int, int] = (1280, 960)) -> list[float]:
    """Error (m) predicting each point from a fit on the others.

    Small per-point errors can hide a badly constrained fit (e.g. most points on one line):
    the fit passes through every click but is wrong elsewhere. This exposes that.
    """
    if len(points) < 5:
        return [float("nan")] * len(points)
    errs = []
    for i, q in enumerate(points):
        calib, _ = fit(points[:i] + points[i + 1:], k1, size)
        if calib is None:
            errs.append(float("inf"))
            continue
        pred = calib.to_pitch([q["pixel"]])[0]
        errs.append(round(float(np.hypot(*(pred - q["pitch"]))), 2))
    return errs


def fit_lens(points: list[dict], size: tuple[int, int]) -> float:
    """Lens parameter k1 that best predicts held-out points (wide-angle lenses bend lines near the edges)."""
    if len(points) < 6:  # too few points to also estimate the lens
        return 0.0
    score = lambda k: float(np.mean(leave_one_out(points, k, size)))
    coarse = min(np.arange(-0.6, 0.61, 0.04), key=score)
    return round(float(min(np.arange(coarse - 0.04, coarse + 0.041, 0.005), key=score)), 3)


def spread_warning(points: list[dict], k1: float = 0.0, size: tuple[int, int] = (1280, 960)) -> str | None:
    """Warn when the clicked points cannot pin down the whole pitch plane."""
    pm = np.array([q["pitch"] for q in points], dtype=np.float32)
    distinct_x, distinct_y = len(np.unique(pm[:, 0].round(1))), len(np.unique(pm[:, 1].round(1)))
    if distinct_x < 2 or distinct_y < 2:
        return "all points lie on one line - add points away from it"
    if len(points) < 5:
        return "only 4 points: no way to check the fit - add at least one more"
    if np.nanmean(leave_one_out(points, k1, size)) > 1.0:
        return "fit is poorly constrained (leave-one-out error > 1 m) - add points in other directions"
    return None


class Clicker:
    def __init__(self, frame: np.ndarray, pitch: Pitch):
        self.frame, self.pitch = frame, pitch
        self.scale = min(1.0, MAX_DISPLAY_WIDTH / frame.shape[1])
        self.todo = landmarks(pitch)
        self.idx, self.points, self.mouse = 0, [], None
        self.canvas = PitchCanvas(pitch, scale=3.0, padding=3.0)

    def on_mouse(self, event, x, y, *_):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and self.idx < len(self.todo):
            name, xy = self.todo[self.idx]
            self.points.append({"name": name, "pixel": [x / self.scale, y / self.scale], "pitch": list(xy),
                                "idx": self.idx})
            self.idx += 1

    def draw(self) -> np.ndarray:
        img = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale)
        for q in self.points:
            p = (int(q["pixel"][0] * self.scale), int(q["pixel"][1] * self.scale))
            cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, q["name"], (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # Mini pitch with the requested landmark highlighted.
        mini = self.canvas.background.copy()
        for q in self.points:
            cv2.circle(mini, tuple(self.canvas.to_px(q["pitch"])), 5, (0, 0, 255), -1)
        if self.idx < len(self.todo):
            cv2.circle(mini, tuple(self.canvas.to_px(self.todo[self.idx][1])), 9, (0, 255, 255), 3)
        h, w = mini.shape[:2]
        img[0:h, img.shape[1] - w:] = mini
        # Magnifier around the cursor.
        if self.mouse:
            mx, my = int(self.mouse[0] / self.scale), int(self.mouse[1] / self.scale)
            pad = cv2.copyMakeBorder(self.frame, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF, cv2.BORDER_CONSTANT)
            crop = pad[my:my + 2 * ZOOM_HALF, mx:mx + 2 * ZOOM_HALF]
            mag = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            c = ZOOM_HALF * ZOOM
            cv2.drawMarker(mag, (c, c), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            mh = mag.shape[0]
            # Keep the magnifier on the side away from the cursor so it never hides what you click.
            x0 = img.shape[1] - mh if self.mouse[0] < img.shape[1] / 2 else 0
            img[img.shape[0] - mh:, x0:x0 + mh] = mag
        status = (f"Click: {self.todo[self.idx][0]}   [s]kip  [u]ndo  [Enter] finish ({len(self.points)} pts)"
                  if self.idx < len(self.todo) else f"All landmarks done - press Enter ({len(self.points)} pts)")
        cv2.rectangle(img, (0, 0), (img.shape[1] - w, 30), (0, 0, 0), -1)
        cv2.putText(img, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def run(self) -> list[dict] | None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.draw())
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                return None
            if key == ord("s") and self.idx < len(self.todo):
                self.idx += 1
            elif key == ord("u") and self.points:
                self.idx = self.points.pop()["idx"]
            elif key in (13, 10) and len(self.points) >= 4:
                return self.points


def review(frame: np.ndarray, pitch: Pitch, points: list[dict], calib: Calibration, errors: list[float],
           scale: float) -> np.ndarray:
    img = frame.copy()
    _draw_pitch_lines(img, calib.H, pitch, calib)
    for q, e in zip(points, errors):
        p = tuple(int(v) for v in q["pixel"])
        cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img, f"{e:.2f} m", (p[0] + 8, p[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    loo = leave_one_out(points, calib.k1, calib.size)
    msg = (f"Pink = pitch through your calibration (lens k1={calib.k1:+.3f}). Error {np.mean(errors):.2f} m, "
           f"leave-one-out {np.nanmean(loo):.2f} m.  [s]ave  [r]edo  [Esc] quit")
    warning = spread_warning(points, calib.k1, calib.size)
    cv2.rectangle(img, (0, 0), (img.shape[1], 56 if warning else 30), (0, 0, 0), -1)
    cv2.putText(img, msg, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if warning:
        cv2.putText(img, "WARNING: " + warning, (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def solve(points: list[dict], size: tuple[int, int]) -> tuple[Calibration | None, list[float]]:
    """Fit lens + homography: lens first (best held-out accuracy), then the homography with it."""
    return fit(points, fit_lens(points, size), size)


def save(out: Path, video: str, frame_idx: int, pitch: Pitch, points: list[dict],
         calib: Calibration, errors: list[float]) -> None:
    points = [{k: v for k, v in q.items() if k in ("name", "pixel", "pitch")} for q in points]
    loo = leave_one_out(points, calib.k1, calib.size)
    warning = spread_warning(points, calib.k1, calib.size)
    out.write_text(json.dumps({
        "video": video, "frame": frame_idx,
        "pitch": {"length": pitch.length, "width": pitch.width},
        "image_size": list(calib.size),
        "k1": calib.k1,
        "points": [q | {"error_m": e, "leave_one_out_m": l} for q, e, l in zip(points, errors, loo)],
        "mean_error_m": float(np.mean(errors)),
        "mean_leave_one_out_m": float(np.nanmean(loo)),
        "warning": warning,
        "homography": calib.H.tolist(),
    }, indent=2))
    print(f"saved {out}  ({len(points)} points, lens k1={calib.k1:+.3f}, error {np.mean(errors):.2f} m, "
          f"leave-one-out {np.nanmean(loo):.2f} m)")
    if warning:
        print("WARNING:", warning)


def read_frame(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"could not read frame {frame_idx} of {video}")
    return frame


def refit(path: Path) -> None:
    """Recompute a saved calibration from its clicked points (no clicking), e.g. to add the lens model."""
    d = json.loads(path.read_text())
    pitch = Pitch(length=d["pitch"]["length"], width=d["pitch"]["width"])
    frame = read_frame(Path(d["video"]), d["frame"])
    size = (frame.shape[1], frame.shape[0])
    calib, errors = solve(d["points"], size)
    save(path, d["video"], d["frame"], pitch, d["points"], calib, errors)
    preview = path.with_suffix(".png")
    cv2.imwrite(str(preview), review(frame, pitch, d["points"], calib, errors, 1.0))
    print(f"preview -> {preview}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path, nargs="?")
    p.add_argument("--frame", type=int, default=0, help="frame to calibrate on")
    p.add_argument("--length", type=float, default=105.0, help="pitch length (m)")
    p.add_argument("--width", type=float, default=68.0, help="pitch width (m)")
    p.add_argument("--out", type=Path, default=None, help="default: <video>.calibration.json")
    p.add_argument("--refit", type=Path, nargs="+", default=None,
                   help="recompute saved calibration json(s) from their points, without clicking")
    args = p.parse_args()

    if args.refit:
        for path in args.refit:
            refit(path)
        return
    if args.video is None:
        p.error("give a video to calibrate, or --refit <calibration.json>")

    frame = read_frame(args.video, args.frame)
    size = (frame.shape[1], frame.shape[0])
    pitch = Pitch(length=args.length, width=args.width)
    out = args.out or args.video.with_suffix(".calibration.json")

    while True:
        clicker = Clicker(frame, pitch)
        points = clicker.run()
        if points is None:
            break
        calib, errors = solve(points, size)
        if calib is None:
            print("These points cannot define the pitch (are they all on one line?). Try again.")
            continue
        cv2.imshow(WINDOW, review(frame, pitch, points, calib, errors, clicker.scale))
        key = 0
        while key not in (ord("s"), ord("r"), 27):
            key = cv2.waitKey(0) & 0xFF
        if key == ord("r"):
            continue
        if key == ord("s"):
            save(out, str(args.video), args.frame, pitch, points, calib, errors)
        break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# (foot-cv) rodrigo@DD949NQ6Q7 foot_cv % .venv/bin/python scripts/calibrate_camera.py data/raw/alfheim/2013-11-03_first_0_600s_9s.mp4
# saved data/raw/alfheim/2013-11-03_first_0_600s_9s.calibration.json  (12 points, error 0.89 m, leave-one-out 1.36 m)
# WARNING: fit is poorly constrained (leave-one-out error > 1 m) - add points in other directions