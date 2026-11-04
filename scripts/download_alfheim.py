"""Download a time window of an Alfheim match (Simula) plus its ground-truth player positions.

The video is stored as 3-second raw H.264 chunks; we download the ones covering the window,
join them, and write an H.264 mp4 (optionally downscaled).

Example:
    python scripts/download_alfheim.py --match 2013-11-03 --half first --start 600 --duration 120
"""
import argparse
import json
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
import requests
import truststore

# The server omits an intermediate certificate; verify through the OS trust store, which can fetch it.
truststore.inject_into_ssl()

BASE = "https://datasets.simula.no/downloads/alfheim"
CHUNK_SECONDS = 3
ZXY_FILES = {  # ground-truth positions (sensor data); 2013-11-28 has a single file for the whole match
    "2013-11-03": "2013-11-03_tromso_stromsgodset_raw_{half}.csv",
    "2013-11-07": "2013-11-07_tromso_anji_raw_{half}.csv",
    "2013-11-28": "2013-11-28_tromso_tottenham_raw.csv",
}


def list_chunks(url: str) -> list[tuple[str, pd.Timestamp]]:
    """(url-encoded href, recording time of the chunk's first frame), in time order."""
    html = requests.get(url + "?C=N;O=A", timeout=60).text
    hrefs = sorted(set(re.findall(r'href="\./([^"]+\.h264)"', html)))
    return [(h, pd.Timestamp(urllib.parse.unquote(h)[5:-5])) for h in hrefs]


def camera_url(match: str, half: str, camera: str) -> str:
    if match == "2013-11-28":
        return f"{BASE}/{match}/panorama/"
    return f"{BASE}/{match}/{'First' if half == 'first' else 'Second'}%20Half/{camera}/"


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        tmp.rename(dest)
    return dest


def download_camera(args, camera: str, begin: pd.Timestamp) -> None:
    """Download one camera's chunks covering [begin, begin + duration] and write an mp4."""
    url = camera_url(args.match, args.half, camera)
    chunks = list_chunks(url)
    end = begin + pd.Timedelta(seconds=args.duration)
    step = pd.Timedelta(seconds=CHUNK_SECONDS)
    selected = [(h, ts) for h, ts in chunks if ts < end and ts + step > begin]
    print(f"[{camera}] downloading {len(selected)} of {len(chunks)} chunks")

    cache = args.out / "chunks" / args.match / args.half / camera
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        # Listing hrefs are already URL-encoded; decode only for the local filename.
        files = list(pool.map(lambda c: fetch(url + c[0], cache / urllib.parse.unquote(c[0])), selected))

    name = f"{args.match}_{args.half}_{camera}_{int(args.start)}s_{int(args.duration)}s"
    joined = cache / f"{name}.h264"
    with open(joined, "wb") as out:
        for f in files:
            out.write(f.read_bytes())

    # Raw .h264 has no timing, so OpenCV reports a default 25 fps. The real rate is frames per chunk
    # over the chunk length: 90 frames / 3 s = 30 fps for the single cameras, 75 / 3 = 25 for the panorama.
    probe = cv2.VideoCapture(str(files[0]))
    frames_in_chunk = 0
    while probe.grab():
        frames_in_chunk += 1
    fps = frames_in_chunk / CHUNK_SECONDS

    cap = cv2.VideoCapture(str(joined))
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    size = (args.width, round(h * args.width / w / 2) * 2) if args.width else (w, h)
    target = args.out / f"{name}.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    # Trim to the exact start time; chunk timestamps differ between cameras by up to ~1.5 s.
    skip = round((begin - selected[0][1]).total_seconds() * fps)
    n, total = 0, int(args.duration * fps)
    while ok and n < skip + total:
        if n >= skip:
            writer.write(cv2.resize(frame, size) if args.width else frame)
        ok, frame = cap.read()
        n += 1
    writer.release()
    joined.unlink()
    # Recording time of frame 0, needed to line the video up with the sensor data.
    first_frame = selected[0][1] + pd.Timedelta(seconds=skip / fps)
    target.with_suffix(".json").write_text(json.dumps(
        {"match": args.match, "half": args.half, "camera": camera, "fps": fps,
         "first_frame_time": str(first_frame), "frames": n - skip}, indent=2))
    print(f"[{camera}] video -> {target} ({size[0]}x{size[1]}, {fps:.0f} fps, {n - skip} frames, "
          f"starts {first_frame.time()})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match", default="2013-11-03", choices=list(ZXY_FILES))
    p.add_argument("--half", default="first", choices=["first", "second"])
    p.add_argument("--camera", nargs="+", default=["panorama"], help="one or more of: panorama 0 1 2")
    p.add_argument("--start", type=float, default=0,
                   help="seconds from the start of the half (the panorama's first chunk), same for every camera")
    p.add_argument("--duration", type=float, default=120, help="seconds")
    p.add_argument("--width", type=int, default=0, help="downscale output to this width (0 = keep)")
    p.add_argument("--out", type=Path, default=Path("data/raw/alfheim"))
    args = p.parse_args()

    reference = list_chunks(camera_url(args.match, args.half, "panorama"))[0][1]
    begin = reference + pd.Timedelta(seconds=args.start)
    for camera in args.camera:
        download_camera(args, camera, begin)

    zxy = ZXY_FILES[args.match].format(half=args.half)
    gt = fetch(f"{BASE}/{args.match}/zxy/{zxy}", args.out / zxy)
    print(f"ground truth positions -> {gt}")


if __name__ == "__main__":
    main()
