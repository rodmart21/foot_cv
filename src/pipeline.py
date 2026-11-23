"""Video -> tracking data: detect, track, assign teams, project onto the pitch."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from .homography import Calibration, ViewTransformer
from .pitch import Pitch
from .teams import TeamClassifier, shirt_feature, shirt_pixels

PLAYER_CLASSES = {1: "goalkeeper", 2: "player", 3: "referee"}  # class 0 (ball) comes from the ball model
GOALKEEPER_SHARE = 0.2  # share of a track's detections classed goalkeeper that makes it a goalkeeper

# Duplicate suppression (the detector sometimes boxes one person twice, mostly light kits at night):
NMS_IOU = 0.5          # boxes overlapping more than this are one detection (Ultralytics default 0.7)
NESTED_CONTAIN = 0.6   # a box this much inside another ...
NESTED_FEET = 0.1      # ... with its bottom edge within this fraction of the big box's height ...
NESTED_CENTRE = 0.3    # ... and its centre within this fraction of the big box's width is the same person


def drop_nested_duplicates(xyxy: np.ndarray) -> np.ndarray:
    """Keep-mask removing boxes nested in a larger one that stands on the same spot (same feet).

    A smaller box inside a bigger one with the same bottom edge is the same person boxed twice (e.g.
    without the head). A player standing behind another has his feet higher in the image and is kept.
    """
    keep = np.ones(len(xyxy), dtype=bool)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for big in np.argsort(-area):
        if not keep[big]:
            continue
        b = xyxy[big]
        h, w = b[3] - b[1], b[2] - b[0]
        ix = np.clip(np.minimum(b[2], xyxy[:, 2]) - np.maximum(b[0], xyxy[:, 0]), 0, None)
        iy = np.clip(np.minimum(b[3], xyxy[:, 3]) - np.maximum(b[1], xyxy[:, 1]), 0, None)
        nested = ((ix * iy / np.maximum(area, 1e-6) > NESTED_CONTAIN)
                  & (np.abs(xyxy[:, 3] - b[3]) < NESTED_FEET * h)
                  & (np.abs((xyxy[:, 0] + xyxy[:, 2]) / 2 - (b[0] + b[2]) / 2) < NESTED_CENTRE * w)
                  & (area < area[big]))
        keep &= ~nested
    return keep


@dataclass
class Models:
    players: YOLO
    ball: YOLO
    pitch: YOLO

    @classmethod
    def load(cls, models_dir: Path) -> "Models":
        return cls(
            players=YOLO(models_dir / "football-player-detection.pt"),
            ball=YOLO(models_dir / "football-ball-detection.pt"),
            pitch=YOLO(models_dir / "football-pitch-detection.pt"),
        )


@dataclass
class TrackingResult:
    tracking: pd.DataFrame        # one row per (frame, object)
    homographies: np.ndarray      # (n_frames, 3, 3), NaN where no valid fit yet
    team_colours: dict[int, tuple[int, int, int]]
    fps: float
    calibration: Calibration | None = None   # set when a manual calibration replaced the pitch model


def extract(video: Path, models: Models, device: str, pitch_stride: int = 5,
            conf: float = 0.3, ball_conf: float = 0.2,
            homography: np.ndarray | None = None) -> tuple[pd.DataFrame, list, np.ndarray, float]:
    """Single inference pass over the video. Returns raw rows, shirt features, homographies, fps.

    homography: fixed image->pitch transform from a manual calibration; skips the pitch model.
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    view = ViewTransformer(Pitch().vertices)
    view.H = homography
    rows, features, homographies = [], [], []

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        if homography is None and frame_idx % pitch_stride == 0:
            kp = models.pitch.predict(frame, conf=conf, device=device, verbose=False)[0].keypoints
            if kp is not None and len(kp.xy):
                view.update(kp.xy[0].cpu().numpy(), kp.conf[0].cpu().numpy())
        homographies.append(view.H if view.H is not None else np.full((3, 3), np.nan))

        res = models.players.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=1280,
                                   conf=conf, iou=NMS_IOU, agnostic_nms=True, device=device, verbose=False)[0]
        if res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            keep = np.isin(cls_ids, list(PLAYER_CLASSES))
            keep[keep] = drop_nested_duplicates(xyxy[keep])
            xyxy, cls_ids, ids, confs = xyxy[keep], cls_ids[keep], ids[keep], confs[keep]
            feet = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)
            pitch_xy = view.transform(feet)
            for box, c, tid, cf, (px, py), (mx, my) in zip(xyxy, cls_ids, ids, confs, feet, pitch_xy):
                pixels = shirt_pixels(frame, box)
                shirt_bgr = pixels.mean(0) if len(pixels) else np.full(3, np.nan)
                rows.append(dict(frame=frame_idx, track_id=tid, cls=PLAYER_CLASSES[c], conf=cf,
                                 x1=box[0], y1=box[1], x2=box[2], y2=box[3], px=px, py=py, x_m=mx, y_m=my,
                                 shirt_b=shirt_bgr[0], shirt_g=shirt_bgr[1], shirt_r=shirt_bgr[2]))
                features.append(shirt_feature(pixels))

        ball = models.ball.predict(frame, imgsz=1280, conf=ball_conf, device=device, verbose=False)[0].boxes
        if len(ball):
            i = int(ball.conf.argmax())
            box = ball.xyxy[i].cpu().numpy()
            centre = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
            (mx, my), = view.transform(centre)
            rows.append(dict(frame=frame_idx, track_id=-1, cls="ball", conf=float(ball.conf[i]),
                             x1=box[0], y1=box[1], x2=box[2], y2=box[3], px=centre[0, 0], py=centre[0, 1],
                             x_m=mx, y_m=my))
            features.append(None)

        if frame_idx % 25 == 0:
            print(f"  frame {frame_idx}/{n_frames}", flush=True)

    cap.release()
    df = pd.DataFrame(rows)
    float_cols = df.columns.difference(["frame", "track_id", "cls"])
    df[float_cols] = df[float_cols].astype("float64")
    return df, features, np.array(homographies), fps


def assign_teams(df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, dict]:
    """Cluster outfield players into 2 teams, vote per track, then attach goalkeepers by position."""
    df = df.copy()
    people = df["cls"] != "ball"
    # The detector can flicker between player/goalkeeper/referee: take each track's majority class,
    # except that a track is a goalkeeper once GOALKEEPER_SHARE of its detections say so (keepers are
    # often half-classified as players, while outfield tracks almost never get goalkeeper votes).
    def track_class(s: pd.Series) -> str:
        return "goalkeeper" if (s == "goalkeeper").mean() >= GOALKEEPER_SHARE else s.mode().iat[0]
    df.loc[people, "cls"] = df[people].groupby("track_id")["cls"].transform(track_class)

    feat_ok = np.array([f is not None for f in features])
    players = (df["cls"] == "player").to_numpy() & feat_ok
    X = np.stack([features[i] for i in np.flatnonzero(players)])
    labels = TeamClassifier().fit(X).predict(X)
    df["team"] = -1
    df.loc[players, "team"] = labels
    df["team_raw"] = df["team"]  # per-detection vote, kept to measure team consistency
    votes = df[players].groupby("track_id")["team"].agg(lambda s: s.mode().iat[0])
    is_player = df["cls"] == "player"
    df.loc[is_player, "team"] = df.loc[is_player, "track_id"].map(votes).fillna(-1).astype(int)

    # Goalkeepers: nearest team centroid (in metres) per frame, then majority per track.
    centroids = df[is_player & (df["team"] >= 0)].groupby(["frame", "team"])[["x_m", "y_m"]].mean()
    gk = df["cls"] == "goalkeeper"
    for idx, row in df[gk].iterrows():
        try:
            c = centroids.loc[row["frame"]]
            df.at[idx, "team"] = int(((c - row[["x_m", "y_m"]].astype(float)) ** 2).sum(1).idxmin())
        except (KeyError, ValueError):
            pass
    if gk.any():
        gk_votes = df[gk].groupby("track_id")["team"].agg(lambda s: s.mode().iat[0])
        df.loc[gk, "team"] = df.loc[gk, "track_id"].map(gk_votes)

    colours = (df[is_player & (df["team"] >= 0)].groupby("team")[["shirt_b", "shirt_g", "shirt_r"]]
               .median().round().astype(int))
    team_colours = {int(t): tuple(int(v) for v in row) for t, row in colours.iterrows()}
    return df, team_colours


def postprocess(df: pd.DataFrame, fps: float, max_ball_speed: float = 40.0, max_gap: int = 12) -> pd.DataFrame:
    """Smooth player positions and clean/interpolate the ball trajectory."""
    df = df.sort_values(["track_id", "frame"]).copy()
    people = df["cls"] != "ball"
    for col in ("x_m", "y_m"):
        df.loc[people, col] = (df[people].groupby("track_id")[col]
                               .transform(lambda s: s.rolling(5, center=True, min_periods=1).median()))

    ball = df[~people].set_index("frame").sort_index()
    # Drop detections implying an impossible ball speed (false positives: heads, socks, line marks).
    for _ in range(3):
        step = np.hypot(ball["x_m"].diff(), ball["y_m"].diff()) / (ball.index.to_series().diff() / fps)
        bad = step > max_ball_speed
        if not bad.any():
            break
        ball = ball[~bad]
    full = ball.reindex(range(ball.index.min(), ball.index.max() + 1)) if len(ball) else ball
    interp_cols = ["x1", "y1", "x2", "y2", "px", "py", "x_m", "y_m"]
    full["interpolated"] = full["cls"].isna()
    full[interp_cols] = full[interp_cols].interpolate(limit=max_gap, limit_area="inside")
    full = full.dropna(subset=["px"]).assign(cls="ball", track_id=-1, team=-1)
    full = full.reset_index().rename(columns={"index": "frame"})

    out = pd.concat([df[people].assign(interpolated=False), full], ignore_index=True)
    out["t"] = out["frame"] / fps
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)


def recompute_shirt_features(video: Path, raw: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Shirt colour features for every cached detection, re-read from the video with the current
    teams.py settings, so changing the colour method never needs the detection models rerun."""
    raw = raw.copy()
    features = [None] * len(raw)
    people = raw.index[raw.cls != "ball"]
    by_frame = raw.loc[people].groupby("frame").groups
    cap = cv2.VideoCapture(str(video))
    for frame_idx in range(int(raw.frame.max()) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        for i in by_frame.get(frame_idx, []):
            pixels = shirt_pixels(frame, raw.loc[i, ["x1", "y1", "x2", "y2"]].to_numpy(float))
            bgr = pixels.mean(0) if len(pixels) else np.full(3, np.nan)
            raw.loc[i, ["shirt_b", "shirt_g", "shirt_r"]] = bgr
            features[raw.index.get_loc(i)] = shirt_feature(pixels)
    return raw, features


def reproject(raw: pd.DataFrame, calibration: Calibration) -> pd.DataFrame:
    """Recompute pitch positions from the stored pixel anchors with a fixed calibration."""
    raw = raw.copy()
    raw[["x_m", "y_m"]] = calibration.to_pitch(raw[["px", "py"]].to_numpy()).astype("float64")
    return raw


def run(video: Path, models_dir: Path, device: str, pitch_stride: int = 5,
        cache_dir: Path | None = None, calibration: Calibration | None = None) -> TrackingResult:
    """cache_dir: if set, raw inference output is stored there and reused on the next run.
    calibration: fixed camera calibration; positions are recomputed from pixels, so a cache made
    without it is reused.
    """
    homography = calibration.H if calibration is not None else None
    cache = cache_dir / "raw_inference.pkl" if cache_dir else None
    if cache and cache.exists():
        print(f"1/3 reusing cached inference from {cache} (shirt colours re-read from the video)")
        raw, _, homographies, fps = pd.read_pickle(cache)
        raw, features = recompute_shirt_features(video, raw)
    else:
        print("1/3 detection + tracking" + (" (fixed calibration)" if homography is not None else " + pitch keypoints"))
        raw, features, homographies, fps = extract(video, Models.load(models_dir), device, pitch_stride,
                                                   homography=homography)
        if cache:
            pd.to_pickle((raw, features, homographies, fps), cache)
    if calibration is not None:
        raw = reproject(raw, calibration)
        homographies = np.repeat(homography[None], len(homographies), axis=0)
    print("2/3 team assignment")
    df, team_colours = assign_teams(raw, features)
    print("3/3 smoothing + ball cleanup")
    df = postprocess(df, fps)
    cols = ["frame", "t", "track_id", "cls", "team", "team_raw", "conf", "x_m", "y_m", "px", "py",
            "x1", "y1", "x2", "y2", "interpolated"]
    return TrackingResult(df[cols], homographies, team_colours, fps, calibration)
