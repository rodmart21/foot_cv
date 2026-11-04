"""Quality report for a pipeline output dir: metrics, charts and snapshots of problem moments."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diagnostics import Report, diagnose


def plot_overview(report: Report, tracking: pd.DataFrame, fps: float, path: Path) -> None:
    pf = report.per_frame
    t = pf.index / fps
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax[0].plot(t, pf.people, lw=0.8, label="people detected")
    ax[0].plot(t, pf.team_0, lw=0.8, label="team 0")
    ax[0].plot(t, pf.team_1, lw=0.8, label="team 1")
    ax[0].fill_between(t, 0, pf.people.max() + 2, where=~pf.homography.astype(bool),
                       color="red", alpha=0.15, label="no pitch mapping")
    ax[0].set_ylabel("count"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Detections per frame")

    # Track timeline: one horizontal line per track ID; breaks show as gaps / short segments.
    people = tracking[tracking.cls != "ball"]
    order = people.groupby("track_id")["frame"].min().sort_values().index
    colours = {0: "tab:orange", 1: "tab:green", -1: "grey"}
    for i, tid in enumerate(order):
        tr = people[people.track_id == tid]
        ax[1].scatter(tr.frame / fps, np.full(len(tr), i), s=0.5,
                      c=colours.get(int(tr.team.iloc[0]), "grey"), marker="|")
    for ev in report.overlaps.itertuples():
        if ev.outcome != "ok":
            ax[1].axvline(ev.start / fps, color="red", lw=0.5, alpha=0.5)
    ax[1].set_ylabel("track (sorted by start)")
    ax[1].set_title(f"Track timeline — {len(order)} IDs (red lines = overlaps that lost/swapped an ID)")

    ball = pf.ball_detected.astype(int) + 0.5 * pf.ball_interpolated.astype(int)
    ax[2].fill_between(t, 0, ball, step="mid", color="tab:blue", alpha=0.6)
    ax[2].set_yticks([0, 0.5, 1], ["missing", "interpolated", "detected"])
    ax[2].set_xlabel("time (s)"); ax[2].set_title("Ball")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()


def plot_speeds(tracking: pd.DataFrame, fps: float, path: Path) -> None:
    from src.diagnostics import MAX_HUMAN_SPEED, SPRINT_SPEED, add_speed
    p = add_speed(tracking[tracking.cls == "player"], fps)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(p.speed.dropna().clip(upper=20), bins=80, color="tab:blue")
    ax.axvline(SPRINT_SPEED, color="orange", ls="--", label=f"sprint ({SPRINT_SPEED} m/s)")
    ax.axvline(MAX_HUMAN_SPEED, color="red", ls="--", label=f"impossible ({MAX_HUMAN_SPEED} m/s)")
    ax.set_yscale("log"); ax.set_xlabel("player speed (m/s, clipped at 20)"); ax.legend()
    ax.set_title("Speed distribution — mass to the right of the red line = tracking/mapping errors")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()


def snapshots(video: Path, tracking: pd.DataFrame, events: list[tuple[int, list[int], str]],
              path: Path, crop: int = 360) -> None:
    """Grid of crops from the annotated video, centred on each problem moment."""
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for frame, ids, title in events:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            continue
        rows = tracking[(tracking.frame.between(frame - 3, frame + 3)) & tracking.track_id.isin(ids)]
        cx, cy = (int(rows.px.mean()), int(rows.py.mean())) if len(rows) else (img.shape[1] // 2, img.shape[0] // 2)
        x0 = int(np.clip(cx - crop // 2, 0, img.shape[1] - crop))
        y0 = int(np.clip(cy - crop // 2, 0, img.shape[0] - crop))
        tile = img[y0:y0 + crop, x0:x0 + crop].copy()
        cv2.rectangle(tile, (0, 0), (crop, 26), (0, 0, 0), -1)
        cv2.putText(tile, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cap.release()
    if not tiles:
        return
    cols = 4
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), grid)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path, help="pipeline output dir, e.g. outputs/stuttgart_10s")
    p.add_argument("--max-snapshots", type=int, default=12)
    args = p.parse_args()

    d = args.out_dir
    meta = json.loads((d / "meta.json").read_text())
    tracking = pd.read_parquet(d / "tracking.parquet")
    homographies = np.load(d / "homographies.npy")
    cap = cv2.VideoCapture(meta["video"])
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    fps = meta["fps"]

    report = diagnose(tracking, homographies, fps, frame_size)
    diag = d / "diagnostics"
    diag.mkdir(exist_ok=True)
    (diag / "summary.json").write_text(json.dumps(report.summary, indent=2))
    for name in ("overlaps", "teleports", "sprints", "breaks"):
        getattr(report, name).to_csv(diag / f"{name}.csv", index=False)
    plot_overview(report, tracking, fps, diag / "overview.png")
    plot_speeds(tracking, fps, diag / "speeds.png")

    # Problem moments: failed overlaps first, then teleports, then mid-image track breaks.
    events = [(int((e.start + e.end) // 2), [e.id_a, e.id_b], f"t={e.start / fps:.1f}s overlap {e.id_a}/{e.id_b}: {e.outcome}")
              for e in report.overlaps[report.overlaps.outcome != "ok"].itertuples()]
    tele = report.teleports.sort_values("step_speed", ascending=False).drop_duplicates("track_id")
    events += [(int(e.frame), [e.track_id], f"t={e.frame / fps:.1f}s jump id {e.track_id}: {e.step_speed / fps:.1f} m")
               for e in tele.itertuples()]
    events += [(int(e.frame), [e.track_id], f"t={e.frame / fps:.1f}s track {e.track_id} {e.kind}s mid-image")
               for e in report.breaks.itertuples()]
    snapshots(d / "annotated.mp4", tracking, events[:args.max_snapshots], diag / "problem_moments.jpg")
    # Sprints that the tracker held on to, as a sanity check of the good case.
    good = [(int((s.start + s.end) // 2), [s.track_id], f"t={s.start / fps:.1f}s sprint id {s.track_id} {s.max_speed:.1f} m/s")
            for s in report.sprints[report.sprints.track_survives.astype(bool)].sort_values("max_speed", ascending=False).itertuples()]
    snapshots(d / "annotated.mp4", tracking, good[:8], diag / "sprints.jpg")

    print(json.dumps(report.summary, indent=2))
    print(f"-> {diag}")


if __name__ == "__main__":
    main()
