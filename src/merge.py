"""Merge per-camera tracking outputs of the same clip into one table.

Assumes the cameras are frame-synchronised (same frame index = same moment) and calibrated to the
same pitch coordinates. Steps:
    1. pair detections seen by two cameras in the same frame (within GATE metres)
    2. align team labels: flip a camera's 0/1 labels when its pairs disagree with the reference camera
    3. link track IDs across cameras that are paired in many frames (one player = one merged ID)
    4. resolve team disagreements by a majority vote over each merged ID
    5. deduplicate: one row per player per frame, position weighted towards the camera that sees
       him closest to its image centre (least lens distortion)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

GATE = 2.0           # m; two cameras' detections closer than this in the same frame are one player
MIN_LINK_FRAMES = 5  # frames two tracks must be paired before they are treated as the same player


def load_views(dirs: list[Path]) -> tuple[pd.DataFrame, dict]:
    """All cameras' rows in one table with a `camera` column and a camera-unique track id `uid`."""
    frames, metas = [], {}
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        cam = json.loads(Path(meta["video"]).with_suffix(".json").read_text())["camera"]
        t = pd.read_parquet(d / "tracking.parquet").assign(camera=cam)
        t["uid"] = cam + ":" + t.track_id.astype(str)
        frames.append(t)
        metas[cam] = meta
    return pd.concat(frames, ignore_index=True), metas


def pair_detections(people: pd.DataFrame, cam_a: str, cam_b: str) -> pd.DataFrame:
    """Per frame, optimal one-to-one pairing of camera A and B detections within GATE."""
    a_by_frame = dict(tuple(people[people.camera == cam_a].groupby("frame")))
    b_by_frame = dict(tuple(people[people.camera == cam_b].groupby("frame")))
    rows = []
    for f in a_by_frame.keys() & b_by_frame.keys():
        a, b = a_by_frame[f], b_by_frame[f]
        d = np.hypot(a.x_m.to_numpy()[:, None] - b.x_m.to_numpy()[None],
                     a.y_m.to_numpy()[:, None] - b.y_m.to_numpy()[None])
        r, c = linear_sum_assignment(d)
        ok = d[r, c] < GATE
        rows.append(pd.DataFrame({"frame": f, "idx_a": a.index[r[ok]], "idx_b": b.index[c[ok]], "dist": d[r, c][ok]}))
    cols = ["frame", "idx_a", "idx_b", "dist"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def align_teams(people: pd.DataFrame, pairs: dict, reference: str) -> tuple[dict, dict]:
    """Decide per camera whether to flip its team labels so they match the reference camera.

    Walks outwards from the reference through camera pairs that overlap. Returns {camera: flipped}
    and per-pair statistics (agreement before/after).
    """
    flipped, stats = {reference: False}, {}
    frontier = [reference]
    while frontier:
        known = frontier.pop()
        for (a, b), p in pairs.items():
            if known not in (a, b) or (a in flipped and b in flipped):
                continue
            other = b if known == a else a
            ta = people.loc[p.idx_a, "team"].to_numpy()
            tb = people.loc[p.idx_b, "team"].to_numpy()
            valid = (ta >= 0) & (tb >= 0)
            same = float((ta[valid] == tb[valid]).mean()) if valid.any() else np.nan
            flip_needed = bool(same < 0.5)
            flipped[other] = flipped[known] ^ flip_needed
            stats[f"{a}-{b}"] = {"pairs_with_team": int(valid.sum()), "labels_swapped": flip_needed,
                                 "agree_before_alignment": round(same, 3),
                                 "agree_after_alignment": round(max(same, 1 - same), 3)}
            frontier.append(other)
    return flipped, stats


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def link_tracks(people: pd.DataFrame, pairs: dict) -> tuple[pd.Series, dict]:
    """Merged player id per row: tracks paired across cameras in >= MIN_LINK_FRAMES frames are one player.

    Links are applied strongest first, with a cannot-link rule: two groups are only joined if they never
    appear in the same camera in the same frame (one player cannot be two detections in one image).
    Without it, links chain (A=B, B=C => A=C) and glue neighbouring players together.
    """
    candidates = []
    for p in pairs.values():
        counts = pd.DataFrame({"a": people.loc[p.idx_a, "uid"].to_numpy(),
                               "b": people.loc[p.idx_b, "uid"].to_numpy()}).value_counts()
        candidates += [(n, a, b) for (a, b), n in counts.items() if n >= MIN_LINK_FRAMES]
    candidates.sort(reverse=True)

    uf = UnionFind()
    # per group: set of (camera, frame) it occupies
    occupied = {uid: set(zip(g.camera, g.frame)) for uid, g in people.groupby("uid")}
    accepted = rejected = 0
    for _, a, b in candidates:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        if occupied[ra] & occupied[rb]:
            rejected += 1
            continue
        uf.union(ra, rb)
        occupied[rb] |= occupied.pop(ra)
        accepted += 1
    roots = people.uid.map(uf.find)
    return roots.astype("category").cat.codes, {"candidates": len(candidates), "accepted": accepted,
                                                 "rejected_by_cannot_link": rejected}


def resolve_teams(people: pd.DataFrame) -> pd.Series:
    """One team per merged player: majority of his rows (referees keep -1)."""
    voted = people[people.team >= 0].groupby("merged_id").team.agg(lambda s: s.mode().iat[0])
    return people.merged_id.map(voted).fillna(-1).astype(int).where(people.cls != "referee", -1)


def deduplicate(people: pd.DataFrame, pairs: dict, image_size: tuple[int, int]) -> pd.DataFrame:
    """One row per player per frame. Rows of the same merged player, or paired in that frame, collapse;
    the position is weighted towards the camera that sees the player nearest its image centre."""
    uf = UnionFind()
    for idx in people.index:
        uf.find(idx)
    first_row = people.reset_index().groupby(["frame", "merged_id"])["index"].first()
    for (f, mid), idx in people.reset_index().groupby(["frame", "merged_id"])["index"]:
        for i in idx:
            uf.union(i, first_row[(f, mid)])
    for p in pairs.values():
        for a, b in zip(p.idx_a, p.idx_b):
            uf.union(a, b)
    people = people.assign(cluster=[uf.find(i) for i in people.index])

    w, h = image_size
    r2 = ((people.px - w / 2) / (w / 2)) ** 2 + ((people.py - h / 2) / (h / 2)) ** 2
    people = people.assign(weight=1.0 / (0.1 + r2))

    def collapse(g: pd.DataFrame) -> pd.Series:
        wsum = g.weight.sum()
        main = g.merged_id.value_counts().index[0]
        return pd.Series({
            "frame": g.frame.iat[0], "t": g.t.iat[0], "track_id": main,
            "cls": g.cls.mode().iat[0], "team": g.loc[g.merged_id == main, "team"].iat[0],
            "conf": g.conf.max(),
            "x_m": (g.x_m * g.weight).sum() / wsum, "y_m": (g.y_m * g.weight).sum() / wsum,
            "camera": "+".join(sorted(g.camera.unique())), "n_cameras": g.camera.nunique(),
            "source_ids": ",".join(sorted(g.uid.unique())),
        })

    return people.groupby("cluster", sort=False).apply(collapse).reset_index(drop=True)


def merge_ball(balls: pd.DataFrame) -> pd.DataFrame:
    """One ball per frame: prefer real detections over interpolated ones, then highest confidence."""
    best = balls.sort_values(["interpolated", "conf"], ascending=[True, False]).groupby("frame").head(1)
    return best.assign(n_cameras=1, source_ids="", track_id=-1)[
        ["frame", "t", "track_id", "cls", "team", "conf", "x_m", "y_m", "camera", "n_cameras", "source_ids"]]


def merge(dirs: list[Path], reference: str = "1", image_size: tuple[int, int] = (1280, 960)
          ) -> tuple[pd.DataFrame, dict, dict]:
    """Merged tracking table, merge report and merged team colours (BGR)."""
    rows, metas = load_views(dirs)
    people = rows[rows.cls != "ball"].dropna(subset=["x_m"]).copy()
    cams = sorted(people.camera.unique())
    pairs = {(a, b): pair_detections(people, a, b) for i, a in enumerate(cams) for b in cams[i + 1:]}
    pairs = {k: v for k, v in pairs.items() if len(v)}

    # 2. team labels in the reference camera's convention
    flipped, team_stats = align_teams(people, pairs, reference)
    for cam, flip in flipped.items():
        if flip:
            sel = (people.camera == cam) & (people.team >= 0)
            people.loc[sel, "team"] = 1 - people.loc[sel, "team"]
    colours = {}
    for team in (0, 1):
        bgr = [metas[c]["team_colours_bgr"][str(1 - team if flipped.get(c) else team)] for c in metas
               if str(1 - team if flipped.get(c) else team) in metas[c]["team_colours_bgr"]]
        colours[str(team)] = np.round(np.mean(bgr, axis=0)).astype(int).tolist()

    # 3-4. one id and one team per player across cameras
    people["merged_id"], links = link_tracks(people, pairs)
    before = {k: float((people.loc[p.idx_a, "team"].to_numpy() != people.loc[p.idx_b, "team"].to_numpy()).mean())
              for k, p in pairs.items()}
    people["team"] = resolve_teams(people)
    after = {k: float((people.loc[p.idx_a, "team"].to_numpy() != people.loc[p.idx_b, "team"].to_numpy()).mean())
             for k, p in pairs.items()}

    # 5. one row per player per frame
    merged = deduplicate(people, pairs, image_size)
    ball = merge_ball(rows[rows.cls == "ball"])
    out = pd.concat([merged, ball], ignore_index=True).sort_values(["frame", "track_id"]).reset_index(drop=True)

    report = {
        "cameras": cams, "reference_camera": reference, "labels_flipped": flipped,
        "team_alignment": team_stats,
        "duplicates_per_frame": {f"{a}-{b}": round(len(p) / people.frame.nunique(), 2) for (a, b), p in pairs.items()},
        "median_gap_between_cameras_m": {f"{a}-{b}": round(float(p.dist.median()), 2) for (a, b), p in pairs.items()},
        "track_links": links,
        "track_ids_before": int(people.uid.nunique()), "track_ids_after": int(people.merged_id.nunique()),
        "team_disagreement_in_overlap_after_alignment": {f"{a}-{b}": round(v, 3) for (a, b), v in before.items()},
        "team_disagreement_in_overlap_after_resolve": {f"{a}-{b}": round(v, 3) for (a, b), v in after.items()},
        "rows_before": int(len(people)), "rows_after": int(len(merged)),
        "people_per_frame_before": float(people.groupby("frame").size().median()),
        "people_per_frame_after": float(merged.groupby("frame").size().median()),
    }
    return out, report, colours
