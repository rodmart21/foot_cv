"""Run the full pipeline on a video: tracking data (parquet) + annotated video."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from foot_cv.pipeline import run
from foot_cv.render import render


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=None, help="output dir (default: outputs/<video stem>)")
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--pitch-stride", type=int, default=5, help="re-estimate the homography every N frames")
    p.add_argument("--no-video", action="store_true", help="skip rendering the annotated video")
    p.add_argument("--fresh", action="store_true", help="ignore cached inference and rerun the models")
    args = p.parse_args()

    out = args.out or Path("outputs") / args.video.stem
    out.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        (out / "raw_inference.pkl").unlink(missing_ok=True)
    start = time.time()
    result = run(args.video, args.models, args.device, args.pitch_stride, cache_dir=out)
    result.tracking.to_parquet(out / "tracking.parquet", index=False)
    np.save(out / "homographies.npy", result.homographies)
    (out / "meta.json").write_text(json.dumps(
        {"video": str(args.video), "fps": result.fps, "team_colours_bgr": result.team_colours}, indent=2))
    print(f"tracking done in {time.time() - start:.0f}s -> {out}")

    if not args.no_video:
        render(args.video, result, out / "annotated.mp4")
        print(f"video -> {out / 'annotated.mp4'}")


if __name__ == "__main__":
    main()
