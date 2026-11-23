"""Merge the per-camera pipeline outputs of one clip into a single tracking table.

    python scripts/merge_cameras.py outputs/tromso_600s/2013-11-03_first_{0,1,2}_600s_9s

Writes <match>_<half>_merged_<clip>/ next to the inputs: tracking.parquet, meta.json, merge_report.json.
"""
import argparse
import json
from pathlib import Path

from src.merge import merge
from src.stitch import join_tracks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path, help="pipeline output dirs of the same clip, one per camera")
    p.add_argument("--reference", default="1", help="camera whose team labels the others are aligned to")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--join-gap", type=float, default=2.0,
                   help="join track pieces of the same player separated by up to this many seconds (0 = off)")
    args = p.parse_args()

    tracking, report, colours = merge(args.dirs, args.reference)
    if args.join_gap > 0:
        fps = json.loads((args.dirs[0] / "meta.json").read_text())["fps"]
        tracking, report["track_joining"] = join_tracks(tracking, fps, args.join_gap)

    # e.g. 2013-11-03_first_0_600s_9s -> 2013-11-03_first_merged_600s_9s
    parts = args.dirs[0].name.split("_")
    out = args.out or args.dirs[0].parent / "_".join(parts[:2] + ["merged"] + parts[3:])
    out.mkdir(parents=True, exist_ok=True)
    tracking.to_parquet(out / "tracking.parquet", index=False)
    ref_meta = json.loads((args.dirs[[d.name.split("_")[2] for d in args.dirs].index(args.reference)]
                           / "meta.json").read_text())
    (out / "meta.json").write_text(json.dumps({
        "camera": "merged",
        "video": ref_meta["video"],   # cameras are synchronised: the reference clip's timing applies
        "fps": ref_meta["fps"],
        "team_colours_bgr": colours,
        "sources": [str(d) for d in args.dirs],
    }, indent=2))
    (out / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
