"""Append the improvement-plan metrics for one clip to a CSV, so each batch can be compared.

Run after the pipeline, merge_cameras.py and evaluate_alfheim.py on the clip:
    python scripts/log_metrics.py "baseline" --clip 600s_9s

Targets (set before batch A): camera 0 Tromsø detections <= 11 per frame, merged 24-26 people per frame,
found within 2 m >= 74%, team correct >= 93% (>= 92% in the middle third), main-ID share >= 75%.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_alfheim import load_gt

OUT = Path("outputs/tromso_600s")
GT = Path("data/raw/alfheim/2013-11-03_tromso_stromsgodset_raw_first.csv")
OFFSET = 3.4


def nested_pairs_per_frame(t: pd.DataFrame) -> float:
    """Pairs of boxes in the same frame where one is more than 60% inside the other."""
    p, count = t[t.cls != "ball"], 0
    for _, g in p.groupby("frame"):
        b = g[["x1", "y1", "x2", "y2"]].to_numpy()
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        x1, y1 = np.maximum(b[:, None, 0], b[None, :, 0]), np.maximum(b[:, None, 1], b[None, :, 1])
        x2, y2 = np.minimum(b[:, None, 2], b[None, :, 2]), np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        contain = inter / np.minimum(area[:, None], area[None])
        count += int((np.triu(contain, 1) > 0.6).sum())
    return count / p.frame.nunique()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("label", help="name of this run, e.g. 'baseline' or 'batch A'")
    p.add_argument("--clip", default="600s_9s")
    p.add_argument("--match", default="2013-11-03_first")
    p.add_argument("--update", action="store_true", help="overwrite the row with this label instead of appending")
    p.add_argument("--root", type=Path, default=OUT, help="folder holding the clip's output dirs")
    args = p.parse_args()

    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "label": args.label, "clip": args.clip}
    for cam in ("0", "1"):
        d = args.root / f"{args.match}_{cam}_{args.clip}"
        t = pd.read_parquet(d / "tracking.parquet")
        colours = json.loads((d / "meta.json").read_text())["team_colours_bgr"]
        tromso = int(max(colours, key=lambda k: sum(colours[k])))   # lighter kit
        people = t[t.cls != "ball"]
        row[f"cam{cam}_tromso_per_frame"] = round(people[people.team == tromso].groupby("frame").size().mean(), 2)
        row[f"cam{cam}_nested_pairs_per_frame"] = round(nested_pairs_per_frame(t), 2)

    m = args.root / f"{args.match}_merged_{args.clip}"
    merged = pd.read_parquet(m / "tracking.parquet")
    row["merged_people_per_frame"] = float(merged[merged.cls != "ball"].groupby("frame").size().median())
    s = json.loads((m / "evaluation" / "summary.json").read_text())
    row |= {"found_within_2m": s["found_within_2m"], "found_within_2m_by_chance": s["found_within_2m_by_chance"],
            "error_median_m": s["error_median_m"], "team_correct": s["home_team_label_correct"],
            "main_id_share": s["main_id_share_median"],
            "best_id_coverage": s.get("best_id_coverage_median")}
    matches = pd.read_parquet(m / "evaluation" / "matches.parquet").dropna(subset=["team"])
    middle = matches[matches.gy.between(23, 45)]
    home = matches.team.mode()[0]
    row["team_correct_middle_third"] = round(float((middle.team == home).mean()), 3)

    # Opposite error, invisible to team_correct: detections labelled as the sensor team with no sensor
    # player within 4 m, i.e. (almost always) opponents given the home team's label.
    meta = json.loads((m / "meta.json").read_text())
    clip = json.loads(Path(meta["video"]).with_suffix(".json").read_text())
    n = int(merged.frame.max()) + 1
    gt = load_gt(GT, pd.Timestamp(clip["first_frame_time"]), n, clip["fps"], OFFSET, False, True)
    home_rows = merged[(merged.cls != "ball") & (merged.team == home)]
    far = 0
    for f, g in home_rows.groupby("frame"):
        s = gt[gt.frame == f]
        if len(s):
            d = np.hypot(g.x_m.to_numpy()[:, None] - s.gx.to_numpy()[None], g.y_m.to_numpy()[:, None] - s.gy.to_numpy()[None])
            far += int((d.min(1) > 4).sum())
    row["merged_home_label_per_frame"] = round(len(home_rows) / n, 2)
    row["merged_home_label_no_sensor_near"] = round(far / n, 2)

    log = OUT / "metrics_log.csv"   # always the main log, whatever --root is
    df = pd.read_csv(log) if log.exists() else pd.DataFrame()
    if args.update and len(df) and (df.label == args.label).any():
        for k, v in row.items():
            if k != "time":
                df.loc[df.label == args.label, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])])
    df.to_csv(log, index=False)
    print(df.drop(columns=["time"]).set_index("label").T.to_string())


if __name__ == "__main__":
    main()
