"""Score pipeline outputs against Alfheim ground truth (ZXY sensor positions of the home team).

Each output dir must come from a clip made by download_alfheim.py (its .json sidecar gives the
recording time of frame 0). Several dirs are scored separately, and with --merge also combined
(useful for the three single cameras, which each see only part of the pitch).

Example:
    python scripts/evaluate_alfheim.py outputs/2013-11-03_first_{panorama,0,1,2}_600s_9s --merge 0 1 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

L, W = 105.0, 68.0


def load_gt(csv: Path, t0: pd.Timestamp, n_frames: int, fps: float, offset: float,
            flip_x: bool, flip_y: bool) -> pd.DataFrame:
    """Ground truth resampled to video frames, in the pipeline's pitch coordinates."""
    g = pd.read_csv(csv, header=None,
                    names=["ts", "tag", "x", "y", "heading", "direction", "energy", "speed", "dist"])
    g["ts"] = pd.to_datetime(g.ts)
    lo, hi = t0 + pd.Timedelta(seconds=offset - 1), t0 + pd.Timedelta(seconds=offset + n_frames / fps + 1)
    g = g[(g.ts >= lo) & (g.ts <= hi)]
    g["frame"] = ((g.ts - t0).dt.total_seconds() - offset) * fps
    frames = np.arange(n_frames)
    rows = []
    for tag, tr in g.groupby("tag"):
        x = np.interp(frames, tr.frame, tr.x, left=np.nan, right=np.nan)
        y = np.interp(frames, tr.frame, tr.y, left=np.nan, right=np.nan)
        rows.append(pd.DataFrame({"frame": frames, "tag": tag,
                                  "gx": L - x if flip_x else x, "gy": W - y if flip_y else y}))
    gt = pd.concat(rows).dropna()
    return gt[gt.gx.between(-2, L + 2) & gt.gy.between(-2, W + 2)]  # drops substitutes' sensors


def match(gt: pd.DataFrame, people: pd.DataFrame, gate: float) -> pd.DataFrame:
    """Per frame, optimally pair GT players with detections; pairs further apart than `gate` don't count."""
    by_frame = dict(tuple(people.dropna(subset=["x_m"]).groupby("frame")))
    out = []
    for f, gf in gt.groupby("frame"):
        pf = by_frame.get(f)
        res = pd.DataFrame({"frame": f, "tag": gf.tag.to_numpy(), "gx": gf.gx.to_numpy(), "gy": gf.gy.to_numpy(),
                            "track_id": np.nan, "team": np.nan, "x_m": np.nan, "y_m": np.nan, "err": np.nan})
        if pf is not None:
            d = np.hypot(gf.gx.to_numpy()[:, None] - pf.x_m.to_numpy()[None],
                         gf.gy.to_numpy()[:, None] - pf.y_m.to_numpy()[None])
            r, c = linear_sum_assignment(d)
            ok = d[r, c] < gate
            r, c = r[ok], c[ok]
            res.loc[r, ["track_id", "team", "x_m", "y_m"]] = pf.iloc[c][["track_id", "team", "x_m", "y_m"]].to_numpy()
            res.loc[r, "err"] = d[r, c]
        out.append(res)
    return pd.concat(out, ignore_index=True)


TRACKING_GATE = 2.0  # m; per-player tracking stats only count matches this close (well above chance)


def per_player(m2: pd.DataFrame, home: float, fps: float) -> pd.DataFrame:
    """Tracking quality per real (sensor) player, from matches within TRACKING_GATE.

    main_id_share: share of the player's matched time covered by his most common track ID (1.0 = never
    lost); ids_over_10pct: track IDs that cover at least 10% of it (1 = ideal); wrong_team: share of
    matched frames where his detection carried the other team's label.
    """
    rows = []
    for tag, g in m2.groupby("tag"):
        hit = g.dropna(subset=["track_id"])
        ids = hit.track_id.value_counts(normalize=True)
        rows.append({
            "tag": int(tag),
            "seconds_on_pitch": len(g) / fps,
            "share_found_within_2m": len(hit) / len(g),
            "error_median_m": hit.err.median() if len(hit) else np.nan,
            "main_id_share": ids.iloc[0] if len(ids) else np.nan,
            "ids_over_10pct": int((ids >= 0.1).sum()),
            "ids_total": len(ids),
            "wrong_team": float((hit.team != home).mean()) if len(hit) else np.nan,
        })
    return pd.DataFrame(rows).round(3)


def score(gt: pd.DataFrame, people: pd.DataFrame, fps: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Summary, per-player table and frame-by-frame matches (9 m gate).

    Match rates are reported next to a chance level: the same matching against detections taken
    from half a clip away. Players bunch together, so a loose gate matches by luck; only the margin
    over chance reflects real accuracy."""
    n = int(people.frame.max()) + 1
    shifted = people.assign(frame=(people.frame + n // 2) % n)
    m = match(gt, people, 9.0)
    hit = m.dropna(subset=["err"])
    s = {"gt_players_per_frame": float(gt.groupby("frame").size().median())}
    for gate in (2.0, 5.0, 9.0):
        real = match(gt, people, gate).err.notna().mean()
        chance = match(gt, shifted, gate).err.notna().mean()
        s[f"found_within_{gate:.0f}m"] = round(float(real), 3)
        s[f"found_within_{gate:.0f}m_by_chance"] = round(float(chance), 3)
    s |= {
        "error_median_m": round(float(hit.err.median()), 2) if len(hit) else None,
        "error_p90_m": round(float(hit.err.quantile(0.9)), 2) if len(hit) else None,
    }
    players = pd.DataFrame()
    if len(hit) > 20:
        home = hit.team.mode()[0]
        players = per_player(match(gt, people, TRACKING_GATE), home, fps)
        s["home_team_label_correct"] = round(float((hit.team == home).mean()), 3)
        s["main_id_share_median"] = round(float(players.main_id_share.median()), 3)
        s["ids_over_10pct_median"] = float(players.ids_over_10pct.median())
        A = np.c_[hit.gx, hit.gy, np.ones(len(hit))]
        cx = np.linalg.lstsq(A, hit.x_m.astype(float), rcond=None)[0]
        cy = np.linalg.lstsq(A, hit.y_m.astype(float), rcond=None)[0]
        s["fit_x"] = f"x_det = {cx[0]:.2f}*x {cx[1]:+.2f}*y {cx[2]:+.1f}"
        s["fit_y"] = f"y_det = {cy[0]:+.2f}*x {cy[1]:.2f}*y {cy[2]:+.1f}"
    return s, players, m


def find_offset(csv: Path, t0: pd.Timestamp, fps: float, people: pd.DataFrame, flip_x: bool, flip_y: bool,
                coarse: tuple[float, float] = (-30, 30)) -> tuple[pd.DataFrame, float]:
    """Clock offset (sensor minus video, s) where sensor players sit closest to our detections.

    Needs accurate positions (calibrated cameras): the score is the median distance from each sensor
    player to the nearest detection, sampled every 5th frame. Coarse 1 s grid, then 0.1 s around the best.
    """
    from scipy.spatial import cKDTree
    n = int(people.frame.max()) + 1
    frames = range(0, n, 5)
    trees = {f: cKDTree(g[["x_m", "y_m"]].dropna().to_numpy())
             for f, g in people[people.frame.isin(frames)].groupby("frame")}

    def score(offset: float) -> tuple[float, float]:
        gt = load_gt(csv, t0, n, fps, offset, flip_x, flip_y)
        gt = gt[gt.frame.isin(trees.keys())]
        d = np.concatenate([trees[f].query(g[["gx", "gy"]].to_numpy())[0] for f, g in gt.groupby("frame")])
        return float(np.median(d)), float((d < 1.5).mean())

    rows = [(o, *score(o), "coarse") for o in np.arange(coarse[0], coarse[1] + 0.01, 1.0)]
    best = min(rows, key=lambda r: r[1])[0]
    rows += [(round(o, 2), *score(o), "fine") for o in np.arange(best - 1.5, best + 1.51, 0.1)]
    table = pd.DataFrame(rows, columns=["offset_s", "median_dist_m", "within_1_5m", "grid"])
    fine = table[table.grid == "fine"]
    return table, float(fine.loc[fine.median_dist_m.idxmin(), "offset_s"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_dirs", nargs="+", type=Path)
    p.add_argument("--gt", type=Path, default=Path("data/raw/alfheim/2013-11-03_tromso_stromsgodset_raw_first.csv"))
    p.add_argument("--offset", type=float, default=3.4,
                   help="sensor clock minus video clock, seconds (3.4 measured on minutes 3, 10, 25 and 40 "
                        "of the 2013-11-03 first half)")
    p.add_argument("--flip-x", action="store_true")
    p.add_argument("--no-flip-y", dest="flip_y", action="store_false")
    p.add_argument("--merge", nargs="*", default=[], help="also score these views combined (by camera name)")
    p.add_argument("--find-offset", action="store_true",
                   help="search the sensor clock offset using all given dirs together (same clip, calibrated "
                        "cameras); saves <first dir>/evaluation/alignment.csv")
    args = p.parse_args()

    if args.find_offset:
        people, t0, fps = [], None, None
        for d in args.out_dirs:
            meta = json.loads((d / "meta.json").read_text())
            clip = json.loads(Path(meta["video"]).with_suffix(".json").read_text())
            t0, fps = pd.Timestamp(clip["first_frame_time"]), clip.get("fps", meta["fps"])  # clip json has the true frame rate
            t = pd.read_parquet(d / "tracking.parquet")
            people.append(t[t.cls != "ball"])
        table, best = find_offset(args.gt, t0, fps, pd.concat(people), args.flip_x, args.flip_y)
        out = args.out_dirs[0] / "evaluation"
        out.mkdir(exist_ok=True)
        table.to_csv(out / "alignment.csv", index=False)
        b = table[(table.grid == "fine") & (table.offset_s == best)].iloc[0]
        print(f"clip starts {t0} | best offset {best:+.1f} s | median distance {b.median_dist_m:.2f} m | "
              f"within 1.5 m {b.within_1_5m:.0%}  -> {out / 'alignment.csv'}")
        return

    settings = {"offset_s": args.offset, "flip_x": args.flip_x, "flip_y": args.flip_y, "gt": str(args.gt)}

    def save(out: Path, suffix: str, s: dict, players: pd.DataFrame, matches: pd.DataFrame) -> None:
        out.mkdir(exist_ok=True)
        (out / f"summary{suffix}.json").write_text(json.dumps(settings | s, indent=2))
        players.to_csv(out / f"per_player{suffix}.csv", index=False)
        matches.to_parquet(out / f"matches{suffix}.parquet", index=False)
        print(f"saved -> {out}/{{summary,per_player,matches}}{suffix}.*")

    results, views, dirs = {}, {}, {}
    for d in args.out_dirs:
        meta = json.loads((d / "meta.json").read_text())
        clip = json.loads(Path(meta["video"]).with_suffix(".json").read_text())
        t0, fps = pd.Timestamp(clip["first_frame_time"]), clip.get("fps", meta["fps"])  # clip json has the true frame rate
        tracking = pd.read_parquet(d / "tracking.parquet")
        n_frames = int(tracking.frame.max()) + 1
        gt = load_gt(args.gt, t0, n_frames, fps, args.offset, args.flip_x, args.flip_y)
        people = tracking[tracking.cls != "ball"]
        cam = clip["camera"]
        views[cam], dirs[cam] = (gt, people, fps), d
        s, players, matches = score(gt, people, fps)
        save(d / "evaluation", "", s, players, matches)
        results[cam] = s

    if args.merge:
        gt, _, fps = views[args.merge[0]]
        merged = pd.concat([views[c][1].assign(track_id=views[c][1].track_id + 100_000 * i)
                            for i, c in enumerate(args.merge)])
        name = "+".join(args.merge)
        s, players, matches = score(gt, merged, fps)
        save(dirs[args.merge[0]] / "evaluation", f"_merged_{name}", s, players, matches)
        results[f"merged({name})"] = s

    print(f"offset {args.offset:+.1f} s, flip_x={args.flip_x}, flip_y={args.flip_y}")
    print(pd.DataFrame(results).to_string())


if __name__ == "__main__":
    main()
