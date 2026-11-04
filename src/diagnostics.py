"""Quality checks on tracking output when no ground truth is available.

Everything here is a proxy: physically impossible movement, tracks that die mid-pitch,
IDs that do not survive an occlusion. Good tracking data should score near zero on all of them.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

MAX_HUMAN_SPEED = 12.0   # m/s over a 5-frame window; elite sprinters peak ~10.5
TELEPORT_STEP = 1.5      # m in a single frame; box jitter gives ~0.5 m, an ID swap gives several metres
SPRINT_SPEED = 7.0       # m/s
OVERLAP_IOU = 0.2
EDGE_MARGIN = 40         # px; tracks starting/ending this close to the border are fine


@dataclass
class Report:
    summary: dict = field(default_factory=dict)
    overlaps: pd.DataFrame = None
    teleports: pd.DataFrame = None
    sprints: pd.DataFrame = None
    breaks: pd.DataFrame = None
    per_frame: pd.DataFrame = None


def _project(people: pd.DataFrame, homographies: np.ndarray, lag: int) -> np.ndarray:
    """Project each detection's pixel anchor with the homography of frame - lag (metres)."""
    out = np.full((len(people), 2), np.nan)
    frames = people["frame"].to_numpy()
    pts = people[["px", "py"]].to_numpy(np.float32).reshape(-1, 1, 2)
    for f in np.unique(frames):
        H = homographies[max(f - lag, 0)]
        if not np.isnan(H).any():
            idx = np.flatnonzero(frames == f)
            out[idx] = cv2.perspectiveTransform(pts[idx], H).reshape(-1, 2)
    return out


def add_speed(people: pd.DataFrame, fps: float, window: int = 5,
              homographies: np.ndarray | None = None) -> pd.DataFrame:
    """Speed per detection: displacement over `window` frames, plus instantaneous step speed.

    With homographies, step_speed_rel projects this frame's and the previous frame's pixel position
    with the *same* homography, so re-estimating the pitch mapping cannot create fake movement.
    mapping_shift is the part of the raw step caused by the mapping update alone.
    """
    people = people.sort_values(["track_id", "frame"]).copy()
    g = people.groupby("track_id")
    dt = g["frame"].diff() / fps
    people["step_speed"] = np.hypot(g["x_m"].diff(), g["y_m"].diff()) / dt
    if homographies is not None:
        here, prev_h = _project(people, homographies, 0), _project(people, homographies, 1)
        prev_pos = pd.DataFrame(here, index=people.index).groupby(people["track_id"]).shift()
        own = prev_h - prev_pos.to_numpy()
        people["step_speed_rel"] = np.hypot(own[:, 0], own[:, 1]) / dt
        people["mapping_shift"] = np.hypot(*(here - prev_h).T)
    else:
        people["step_speed_rel"] = people["step_speed"]
        people["mapping_shift"] = 0.0
    dtw = g["frame"].diff(window) / fps
    people["speed"] = np.hypot(g["x_m"].diff(window), g["y_m"].diff(window)) / dtw
    return people


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area[:, None] + area[None, :] - inter + 1e-9)


def find_overlaps(people: pd.DataFrame) -> pd.DataFrame:
    """One row per occlusion event: a pair of tracks whose boxes overlap for consecutive frames."""
    hits = []
    for frame, rows in people.groupby("frame"):
        if len(rows) < 2:
            continue
        iou = _iou_matrix(rows[["x1", "y1", "x2", "y2"]].to_numpy())
        ids = rows["track_id"].to_numpy()
        a, b = np.where(np.triu(iou, 1) > OVERLAP_IOU)
        hits += [(frame, min(ids[i], ids[j]), max(ids[i], ids[j]), iou[i, j]) for i, j in zip(a, b)]
    if not hits:
        return pd.DataFrame(columns=["id_a", "id_b", "start", "end", "max_iou"])
    hits = pd.DataFrame(hits, columns=["frame", "id_a", "id_b", "iou"]).sort_values(["id_a", "id_b", "frame"])
    # Consecutive frames (gaps <= 5) of the same pair form one event.
    new_event = (hits.groupby(["id_a", "id_b"])["frame"].diff().fillna(99) > 5).cumsum()
    return (hits.groupby(new_event)
            .agg(id_a=("id_a", "first"), id_b=("id_b", "first"), start=("frame", "min"),
                 end=("frame", "max"), max_iou=("iou", "max"))
            .reset_index(drop=True))


def judge_overlaps(overlaps: pd.DataFrame, people: pd.DataFrame, n_frames: int, fps: float,
                   frame_size: tuple[int, int], after: int = 15) -> pd.DataFrame:
    """Outcome of each occlusion: ok, lost (a track ended right after) or jump (probable ID swap)."""
    if overlaps.empty:
        return overlaps.assign(outcome=[], same_team=[])
    last = people.groupby("track_id")["frame"].max()
    last_row = people.sort_values("frame").groupby("track_id").last()
    team = people.groupby("track_id")["team"].first()
    outcomes = []
    for ev in overlaps.itertuples():
        outcome = "ok"
        for tid in (ev.id_a, ev.id_b):
            near = people[(people.track_id == tid) & people.frame.between(ev.start - 5, ev.end + 5)]
            if (near["step_speed_rel"] / fps > TELEPORT_STEP).any():
                outcome = "jump"
                break
            ends_early = last[tid] < min(ev.end + after, n_frames - 1)
            if ends_early and not _near_edge(last_row.loc[tid], frame_size):
                outcome = "lost"
        outcomes.append(outcome)
    return overlaps.assign(outcome=outcomes,
                           same_team=[team[a] == team[b] for a, b in zip(overlaps.id_a, overlaps.id_b)])


def _near_edge(row, frame_size: tuple[int, int]) -> bool:
    w, h = frame_size
    return row.x1 < EDGE_MARGIN or row.y1 < EDGE_MARGIN or row.x2 > w - EDGE_MARGIN or row.y2 > h - EDGE_MARGIN


def find_breaks(people: pd.DataFrame, n_frames: int, frame_size: tuple[int, int]) -> pd.DataFrame:
    """Tracks that start or end in the middle of the image (not at clip start/end or the border)."""
    rows = []
    srt = people.sort_values("frame")
    for kind, ends, frame_ok in (("start", srt.groupby("track_id").first(), lambda f: f <= 2),
                                 ("end", srt.groupby("track_id").last(), lambda f: f >= n_frames - 3)):
        for tid, r in ends.iterrows():
            if not frame_ok(r.frame) and not _near_edge(r, frame_size):
                rows.append(dict(track_id=tid, kind=kind, frame=int(r.frame), cls=r.cls, x_m=r.x_m, y_m=r.y_m))
    return pd.DataFrame(rows, columns=["track_id", "kind", "frame", "cls", "x_m", "y_m"])


def find_sprints(people: pd.DataFrame, fps: float, min_duration: float = 1.0) -> pd.DataFrame:
    """Sustained runs above SPRINT_SPEED (and below the impossible threshold)."""
    p = people[people.cls == "player"].copy()
    p["fast"] = p["speed"].between(SPRINT_SPEED, MAX_HUMAN_SPEED)
    p["run"] = (p["fast"] != p.groupby("track_id")["fast"].shift()).cumsum()
    runs = (p[p.fast].groupby("run")
            .agg(track_id=("track_id", "first"), start=("frame", "min"), end=("frame", "max"),
                 max_speed=("speed", "max"), team=("team", "first")))
    if runs.empty:
        return pd.DataFrame(columns=["track_id", "start", "end", "max_speed", "team", "track_survives"])
    runs = runs[(runs.end - runs.start) / fps >= min_duration].reset_index(drop=True)
    last = people.groupby("track_id")["frame"].max()
    # A sprint "survives" if the track keeps going at least 1s after the sprint ends.
    runs["track_survives"] = [last[t] >= e + fps or e >= people.frame.max() - fps
                              for t, e in zip(runs.track_id, runs.end)]
    return runs


def diagnose(tracking: pd.DataFrame, homographies: np.ndarray, fps: float,
             frame_size: tuple[int, int]) -> Report:
    n_frames = len(homographies)
    people = add_speed(tracking[tracking.cls != "ball"], fps, homographies=homographies)
    ball = tracking[tracking.cls == "ball"]
    players = people[people.cls == "player"]

    overlaps = judge_overlaps(find_overlaps(people), people, n_frames, fps, frame_size)
    teleports = (people[people.step_speed_rel / fps > TELEPORT_STEP]
                 [["frame", "track_id", "cls", "step_speed_rel", "x_m", "y_m"]]
                 .rename(columns={"step_speed_rel": "step_speed"}))
    # Frames where the whole picture jumped together: a pitch-mapping artefact, not a tracking one.
    shift = people.groupby("frame")["mapping_shift"].median() * fps
    mapping_jumps = int((shift > 5.0).sum())
    sprints = find_sprints(people, fps)
    breaks = find_breaks(people, n_frames, frame_size)

    per_frame = pd.DataFrame(index=pd.RangeIndex(n_frames, name="frame"))
    per_frame["people"] = people.groupby("frame").size()
    for t in (0, 1):
        per_frame[f"team_{t}"] = players[players.team == t].groupby("frame").size()
    per_frame["ball_detected"] = ball[~ball.interpolated].groupby("frame").size() > 0
    per_frame["ball_interpolated"] = ball[ball.interpolated].groupby("frame").size() > 0
    per_frame["homography"] = ~np.isnan(homographies).any(axis=(1, 2))
    per_frame = per_frame.fillna(0).infer_objects()

    track_len = people.groupby("track_id")["frame"].agg(lambda f: f.max() - f.min() + 1) / fps
    voted = players[players.team_raw >= 0]
    team_agreement = float((voted.team == voted.team_raw).mean()) if len(voted) else float("nan")
    minutes = n_frames / fps / 60

    summary = {
        "duration_s": round(n_frames / fps, 1),
        "frames": n_frames,
        "people_per_frame_median": float(per_frame.people.median()),
        "players_per_team_median": [float(per_frame.team_0.median()), float(per_frame.team_1.median())],
        "unique_track_ids": int(people.track_id.nunique()),
        "track_length_median_s": round(float(track_len.median()), 1),
        "tracks_shorter_than_2s": int((track_len < 2).sum()),
        "track_breaks_mid_image": int(len(breaks)),
        "track_breaks_per_min": round(len(breaks) / minutes, 1),
        "team_vote_agreement": round(team_agreement, 3),
        "unassigned_player_detections": int((players.team < 0).sum()),
        "teleports_over_1_5m_per_frame": int(len(teleports)),
        "mapping_jump_frames": mapping_jumps,
        "mapping_jump_frames_pct": round(100 * mapping_jumps / n_frames, 1),
        "overlap_events": int(len(overlaps)),
        "overlap_outcomes": overlaps["outcome"].value_counts().to_dict() if len(overlaps) else {},
        "sprints_over_7ms_1s": int(len(sprints)),
        "sprints_track_survived": int(sprints.track_survives.sum()) if len(sprints) else 0,
        "ball_detected_pct": round(100 * per_frame.ball_detected.mean(), 1),
        "ball_interpolated_pct": round(100 * per_frame.ball_interpolated.mean(), 1),
        "homography_valid_pct": round(100 * per_frame.homography.mean(), 1),
        "positions_off_pitch_pct": round(100 * float(((people.x_m < -3) | (people.x_m > 108) |
                                                      (people.y_m < -3) | (people.y_m > 71)).mean()), 1),
    }
    return Report(summary, overlaps, teleports, sprints, breaks, per_frame)
