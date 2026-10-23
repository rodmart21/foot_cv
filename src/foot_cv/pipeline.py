"""Video -> tracking data: detect, track, assign teams, project onto the pitch."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from foot_cv.homography import ViewTransformer
from foot_cv.pitch import Pitch
from foot_cv.teams import TeamClassifier, shirt_feature, shirt_pixels

PLAYER_CLASSES = {1: "goalkeeper", 2: "player", 3: "referee"}  # class 0 (ball) comes from the ball model


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


def extract(video: Path, models: Models, device: str, pitch_stride: int = 5,
            conf: float = 0.3, ball_conf: float = 0.2) -> tuple[pd.DataFrame, list, np.ndarray, float]:
    """Single inference pass over the video. Returns raw rows, shirt features, homographies, fps."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    view = ViewTransformer(Pitch().vertices)
    rows, features, homographies = [], [], []

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % pitch_stride == 0:
            kp = models.pitch.predict(frame, conf=conf, device=device, verbose=False)[0].keypoints
            if kp is not None and len(kp.xy):
                view.update(kp.xy[0].cpu().numpy(), kp.conf[0].cpu().numpy())
        homographies.append(view.H if view.H is not None else np.full((3, 3), np.nan))

        res = models.players.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=1280,
                                   conf=conf, device=device, verbose=False)[0]
        if res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            keep = np.isin(cls_ids, list(PLAYER_CLASSES))
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
    # The detector can flicker between player/goalkeeper/referee: take each track's majority class.
    df.loc[people, "cls"] = df[people].groupby("track_id")["cls"].transform(lambda s: s.mode().iat[0])

    feat_ok = np.array([f is not None for f in features])
    players = (df["cls"] == "player").to_numpy() & feat_ok
    X = np.stack([features[i] for i in np.flatnonzero(players)])
    labels = TeamClassifier().fit(X).predict(X)
    df["team"] = -1
    df.loc[players, "team"] = labels
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


def run(video: Path, models_dir: Path, device: str, pitch_stride: int = 5,
        cache_dir: Path | None = None) -> TrackingResult:
    """cache_dir: if set, raw inference output is stored there and reused on the next run."""
    cache = cache_dir / "raw_inference.pkl" if cache_dir else None
    if cache and cache.exists():
        print(f"1/3 reusing cached inference from {cache}")
        raw, features, homographies, fps = pd.read_pickle(cache)
    else:
        print("1/3 detection + tracking + pitch keypoints")
        raw, features, homographies, fps = extract(video, Models.load(models_dir), device, pitch_stride)
        if cache:
            pd.to_pickle((raw, features, homographies, fps), cache)
    print("2/3 team assignment")
    df, team_colours = assign_teams(raw, features)
    print("3/3 smoothing + ball cleanup")
    df = postprocess(df, fps)
    cols = ["frame", "t", "track_id", "cls", "team", "conf", "x_m", "y_m", "px", "py",
            "x1", "y1", "x2", "y2", "interpolated"]
    return TrackingResult(df[cols], homographies, team_colours, fps)
