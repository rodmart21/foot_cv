# foot_cv: project status and history

*Last updated: 2026-11-23*

## What this project is

foot_cv turns football match video into **tracking data**: for every frame, where each player, referee and the ball are on the pitch (in metres) and which team each player belongs to. The tracking data is the basis for the end goal: analysing and classifying game situations (team shape, defensive lines, how teams attack and defend).

Design choices:
- **Small vision models, no LLMs** in the pipeline. Everything runs on a laptop GPU (Apple M2) or a cheap cloud GPU.
- **Team-level analysis first.** We track players and their team, but we deliberately **do not recognise individual players** (no shirt numbers or faces).

## Current status (summary)

| Area | Status | Key number (Alfheim, minute-10 clip, 3 cameras merged) |
| --- | --- | --- |
| Pitch mapping | ✅ Solved for fixed cameras (hand calibration + lens correction) | 0.6–0.9 m calibration error |
| Player positions | ✅ Good enough for team-shape analysis | 73% of players within 2 m (15% by chance), median error 1.6 m |
| Team labels | ✅ Good | 97% of home players correct; ~2 opponents per frame still mislabelled |
| Tracking continuity (player IDs) | ⚠️ Weakest part | Main ID covers 66% of a player's time (target 75%) |
| Ball | ❌ Not reliable yet, not checked against ground truth | Detected in ~30–95% of frames depending on clip |
| Validation on more clips | ⏳ Running | Minutes 3, 25, 40 with all 3 cameras, before/after comparison |

Code: branch `final_tracking` (pushed); a PR into `master` is pending.

## How the pipeline works

```
video (per camera)
  → detection        YOLO models: players/goalkeepers/referees, ball (Roboflow pretrained)
  → tracking         ByteTrack: an ID per player, frame to frame
  → team labels      shirt colour histograms, clustered into 2 teams, majority vote per track
  → pitch mapping    pixel → metres, from a fixed hand calibration (or a pitch-landmark model)
  → tracking.parquet one row per object per frame: frame, track_id, class, team, x_m, y_m, ...
then, for multi-camera setups:
  → merge            align team labels between cameras, remove duplicates, link IDs across cameras
  → track joining    stitch pieces of the same player split by short gaps
  → evaluation       against ground-truth sensor positions (Alfheim only)
```

## Repository map

| Path | What it does |
| --- | --- |
| `src/pipeline.py` | Detection, tracking, team assignment, pitch projection, post-processing |
| `src/teams.py` | Shirt colour features and team clustering |
| `src/homography.py` | Pixel → metre mapping: landmark-based, or fixed `Calibration` with lens correction |
| `src/pitch.py` | Pitch geometry (105 × 68 m) and top-down drawing |
| `src/render.py` | Annotated video: team ellipses, ball, projected pitch lines, radar |
| `src/merge.py` | Multi-camera merge |
| `src/stitch.py` | Track joining over time |
| `src/diagnostics.py` | Quality checks that need no ground truth |
| `scripts/run_pipeline.py` | Run the pipeline on a video (`--calibration`, `--fresh`, `--no-video`) |
| `scripts/calibrate_camera.py` | Click tool to calibrate a fixed camera; `--refit` recomputes a saved calibration |
| `scripts/merge_cameras.py` | Merge the per-camera outputs of one clip (`--join-gap`) |
| `scripts/diagnose.py` | Automatic diagnostics (no ground truth) |
| `scripts/evaluate_alfheim.py` | Score against Alfheim sensor data; `--find-offset`, `--merge`, `--snapshots` |
| `scripts/log_metrics.py` | Append a run's key metrics to `outputs/tromso_600s/metrics_log.csv` |
| `scripts/download_alfheim.py` | Download any time window of an Alfheim match, any camera |
| `scripts/make_clip.py` | Cut a clip from a longer video |

`data/` (videos, models, sensor files) and `outputs/` (results) are **not in git**; they live on the machine that runs the pipeline.

## How to run (Alfheim example)

```bash
# 1. Download 10 s of all three cameras, from second 600 of the first half
.venv/bin/python scripts/download_alfheim.py --start 600 --duration 10 --camera 0 1 2

# 2. Calibrate each camera once (click tool; skip if a calibration already exists)
.venv/bin/python scripts/calibrate_camera.py data/raw/alfheim/2013-11-03_first_1_600s_10s.mp4

# 3. Run the pipeline per camera with its calibration
.venv/bin/python scripts/run_pipeline.py data/raw/alfheim/2013-11-03_first_1_600s_10s.mp4 \
    --calibration data/raw/alfheim/2013-11-03_first_1_600s_9s.calibration.json \
    --out outputs/tromso_600s/2013-11-03_first_1_600s_10s

# 4. Merge the three cameras (includes track joining)
.venv/bin/python scripts/merge_cameras.py outputs/tromso_600s/2013-11-03_first_{0,1,2}_600s_10s

# 5. Evaluate against the sensors and log the metrics
.venv/bin/python scripts/evaluate_alfheim.py outputs/tromso_600s/2013-11-03_first_{0,1,2,merged}_600s_10s --snapshots
.venv/bin/python scripts/log_metrics.py "my run" --clip 600s_10s
```

Speed: about 0.6 s per frame per camera on an Apple M2 (three large YOLO models). A 10-second clip takes about 3 minutes per camera; reruns that reuse the cached detections take seconds.

## Project history: what we tried, what we found, how we fixed it

### Phase 1: first pipeline on broadcast footage (Bundesliga, 10 s)

- **Data:** 10 s of a Bundesliga match (Stuttgart), wide broadcast camera, 25 fps.
- **Result:** works well. 23 people per frame, 26 track IDs, teams split 10 / 10, ball found in 91% of frames, pitch lines aligned in the middle of the pitch.
- **Speed:** 137 s for 10 s of video on an M2. A full match would take ~19 h with these models; smaller models and fewer frames per second would bring it to ~30–60 min.
- **Issues found:** the pitch mapping is re-estimated every 5 frames, so all players jump ~0.7 m at once on those frames (makes speeds unreliable). IDs break when players cross.

### Phase 2: a harder, longer clip (Veo youth match, 2 min)

- **Data:** 2 minutes of a youth-academy match filmed with a Veo camera.
- **Result:** fails. Pitch mapping completely wrong (the landmark model doesn't recognise this pitch), teams split 16 / 2 (white vs light blue kits too similar for the colour method), 1,203 track IDs (mostly short noise tracks), ball in 28% of frames.
- **Lesson:** the pipeline only works on footage similar to what its models were trained on. Robustness, not compute, is the main risk for a product. And without ground truth we can't measure accuracy properly.

### Phase 3: Alfheim dataset, a match with ground truth

- **Data:** [Alfheim dataset](https://datasets.simula.no/alfheim/) (Simula), Tromsø IL vs Strømsgodset, 2013-11-03. Three fixed cameras plus a stitched panorama, and **ZXY sensor positions for the home team** (10 outfield players; the goalkeeper and the away team have no sensors). Sensor accuracy is about ±0.2 m (up to ~1 m during sudden movements), per [Medbø & Ylvisåker 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10056700/).
- **Download:** videos come as 3-second raw H.264 chunks; `download_alfheim.py` fetches any time window. The Simula server has an incomplete certificate chain, so the script verifies through the OS trust store (`truststore`).

### Phase 4: the panorama doesn't work

- **Data:** the stitched panorama (4450 × 2000, 25 fps), the whole pitch in one image.
- **Issues:** players are ~15 px tall after the detector shrinks the image to 1280 px; the stitched image bends pitch lines, so one flat pitch mapping can't fit it; ball found in 12% of frames.
- **Key finding:** positions scored **no better than chance** (see "lessons" below).
- **Decision:** use the three individual cameras instead.

### Phase 5: three cameras and hand calibration

- **Data:** cameras 0 (left third), 1 (centre), 2 (right third), 1280 × 960 each.
- **Issue:** the pitch-landmark model fails on these camera angles (projected lines far from the real ones).
- **Fix:** **hand calibration**. A click tool asks for known pitch points (corners, box corners, centre spot); the transform is computed from them. The cameras never move, so one calibration holds for the whole match.
- **Issue found while calibrating:** errors grew towards the image corners. Cause: wide-angle **lens distortion**. Fix: a one-parameter lens model fitted together with the transform.
- **Result:** leave-one-out calibration error 0.90 m (camera 0), 0.74 m (camera 1), 0.58 m (camera 2). Lens parameters k1 = −0.240, −0.070, −0.225.
- **Safety check built in:** the tool warns when clicked points are poorly spread (e.g. all on one line), which fits the clicks perfectly but fails elsewhere.

### Phase 6: lining up video and sensor data

- **Issue 1, chance matching:** with 10 players bunched in one area, almost any detection lands within a few metres of some sensor player. The first accuracy numbers were mostly luck. **Fix:** every score is now reported next to a chance level (the same matching against detections from a different moment).
- **Issue 2, two clocks:** the cameras and the sensors were recorded by different computers. **Finding:** the sensor clock runs **+3.4 s** ahead of the camera clock, constant across the first half (measured at minutes 3, 10, 25 and 40). The sensors' y-axis is also mirrored relative to ours.
- **Issue 3, our own bug:** the individual cameras record at **30 fps**, but OpenCV reported 25 fps for the raw H.264 files. This made the offset look inconsistent between clips. **Fix:** the downloader now measures the frame rate from the frames per chunk; all clips were rebuilt.
- **Also checked:** the three cameras are frame-synchronised (identical chunk timestamps; consistent positions in their overlap).

### Phase 7: merging the three cameras

- **What the merge does:** pairs detections seen by two cameras in the same frame, aligns team labels between cameras (camera 2's labels were swapped), links track IDs across cameras, and removes duplicates.
- **Issue found:** linking tracks naively chained links and glued neighbouring players together (one merged ID was made of 12 tracks). **Fix:** a cannot-link rule (two groups are never joined if they appear in the same camera at the same moment).
- **Result vs single cameras:** players within 2 m 74% (vs 56–61%), team correct 93% (vs 85–91%).

### Phase 8: improvement batches (minute-10 clip)

Issues seen in the merged output: duplicated home players (light kit under green floodlights), too many people per frame, the away goalkeeper with the wrong team, frequent ID breaks.

| Batch | Changes | Effect |
| --- | --- | --- |
| **A** | Stricter, class-agnostic duplicate filter in the detector (overlap 50% instead of 70%); drop boxes nested inside another with the same feet position | Duplicate box pairs halved; team correct 93.3% → 95.7% (middle third 88% → 94%); no real players lost |
| **B** | Same-camera duplicate clean-up in the merge; goalkeeper team by the goal he defends; a track is a goalkeeper from 20% goalkeeper votes (was a majority) | Both goalkeepers on the right team; wrong home labels 4.7 → 3.7 per frame |
| **D** | Tighter shirt-only crop for colour features; shirt colours re-read from video when reusing cached detections | Team correct 95.7% → 97.1%; wrong home labels 3.7 → 3.0 per frame |
| **C** | Track joining across short gaps; merge picks a stable ID (was flipping on ties); cannot-link tolerates up to 3 conflicting frames | Main-ID share 56.6% → 65.9%; best-ID coverage 39% → 47% |

D ran before C: batch B showed mislabelled opponents were the bigger problem, so team colours were moved up.

Tested and rejected for team colours: grey-world white balance against the green cast, a frame-adaptive grass filter, ignoring overlapped crops, average colour instead of a histogram. All were worse or had no effect against the sensors.

**All batches side by side (minute 10, merged, vs sensors):**

| Metric | Baseline | A | B | D | C | Target |
| --- | --- | --- | --- | --- | --- | --- |
| Duplicate box pairs per frame (camera 0) | 1.58 | 0.75 | 0.75 | 0.75 | 0.75 | lower |
| Merged people per frame | 29 | 28 | 28 | 28 | 28 | 24–26 |
| Found within 2 m (chance) | 73.5% (15.7%) | 73.4% | 73.0% | 73.0% | 73.0% (15.3%) | ≥ 74% |
| Median position error | 1.55 m | 1.57 m | 1.58 m | 1.58 m | 1.58 m | – |
| Team correct | 93.3% | 95.7% | 95.7% | 97.1% | 97.2% | ≥ 93% |
| Team correct, middle third | 88.0% | 93.8% | 93.9% | 94.9% | 94.9% | ≥ 92% |
| Wrong home labels per frame | – | 4.7 | 3.7 | 3.0 | 3.0 | lower |
| Main-ID share | 55.6% | 54.6% | 56.6% | 56.6% | 65.9% | ≥ 75% |
| Best-ID coverage (ceiling ~73%) | – | – | – | – | 46.7% | – |

### Phase 9: validation on more clips (in progress)

The gains above were measured on one 9-second clip. The current run adds camera 2 to the minute 3, 25 and 40 clips and compares the code before batches A–D (commit `2da76e9`) with the current code on each clip. Results will be logged in `outputs/tromso_600s/metrics_log.csv` as `before A-D (<clip>)` and `after A-D (<clip>)`.

## Lessons learned

1. **Always compare against chance.** When players bunch together, loose distance checks match by luck. A score only means something next to its chance level.
2. **Check assumptions about the data itself.** The frame rate reported by OpenCV was wrong; two recording systems had different clocks; the sensor axes were mirrored.
3. **Fixed cameras should be calibrated by hand.** The pretrained pitch-landmark model fails outside broadcast-style views; a one-off calibration is exact and free at run time.
4. **Measure what you fix, both directions.** "Team correct" only checked sensor players; the opposite error (opponents labelled as the home team) needed its own metric.
5. **Metrics can be unfair.** The first ID-continuity metric counted crowding as ID breaks; best-ID coverage fixed that.
6. **One clip is not enough.** Improvements must be confirmed on several clips before trusting them.

## Known limitations

- **ID swaps in crowds:** the tracker hands an ID to a neighbouring player when two cross. Track joining can't fix this because both pieces exist at the same moment.
- **Missed players:** ~27% of player-frames have no detection within 2 m (occlusion, fast movement, small far players).
- **Ball:** weak detection, never evaluated against ground truth.
- **Domain dependence:** the detector was trained on broadcast footage; performance drops on different cameras and lighting (night match, green floodlight cast).
- **Evaluation covers the home team only** (the away team and goalkeepers have no sensors).
- **Speed:** ~0.6 s per frame per camera with the current large models; a full match would need smaller models or a cloud GPU.

## Next steps

1. **Finish the validation** on minutes 3, 25 and 40 and confirm the batch gains hold.
2. **Appearance-aware tracking** (e.g. BoT-SORT with re-identification) to reduce ID swaps, now the biggest gap.
3. **Decide on detector fine-tuning** (~200 labelled Alfheim frames) if misses and swaps stay high. It addresses missed players, remaining duplicates and the light kit at night.
4. Optional: **learned appearance features for team labels** (e.g. SigLIP) if the remaining ~2 mislabelled opponents per frame matter.
5. **Full 10-minute run** on the three cameras.
6. **Analysis layer:** team shape (width, depth, compactness), defensive line height, possession phases, then game-situation classification.
7. **Ball:** add a ground-truth check (a few labelled frames) and improve detection.
8. Later, for a product: smaller models, cloud GPU deployment, an upload interface.
