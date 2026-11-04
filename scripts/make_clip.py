"""Cut a short clip out of a longer match video."""
import argparse
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--duration", type=float, default=10.0, help="clip length in seconds")
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))

    args.target.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(args.target), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(int(args.duration * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    out.release()
    print(f"wrote {args.target} ({args.duration}s @ {fps:.0f} fps, {size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
