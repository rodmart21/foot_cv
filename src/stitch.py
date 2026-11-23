"""Track joining: stitch track pieces of the same player that the tracker split (occlusions, crossings).

A piece that ends is joined to a piece that starts shortly after when the move between them is
physically possible, both are on the same team, and they never exist at the same moment. Candidates are
taken cheapest first (closest in space and time) and each piece end/start is used once.
"""
import numpy as np
import pandas as pd

MAX_GAP_S = 2.0      # longest gap between pieces that can still be the same player
MAX_SPEED = 9.0      # m/s; a player cannot cover more than this during the gap ...
MARGIN_M = 1.5       # ... plus this much position noise
EDGE_FRAMES = 3      # start/end position = median of this many frames (less jitter)


def pieces(people: pd.DataFrame) -> pd.DataFrame:
    """One row per track piece: first/last frame and where it starts and ends."""
    people = people.sort_values("frame")
    g = people.groupby("track_id")
    first = g.head(EDGE_FRAMES).groupby("track_id")[["x_m", "y_m"]].median()
    last = g.tail(EDGE_FRAMES).groupby("track_id")[["x_m", "y_m"]].median()
    return pd.DataFrame({
        "start": g.frame.min(), "end": g.frame.max(),
        "x0": first.x_m, "y0": first.y_m, "x1": last.x_m, "y1": last.y_m,
        "team": g.team.agg(lambda s: s.mode().iat[0]), "cls": g.cls.agg(lambda s: s.mode().iat[0]),
    })


def join_tracks(tracking: pd.DataFrame, fps: float, max_gap_s: float = MAX_GAP_S) -> tuple[pd.DataFrame, dict]:
    """Return the table with joined track ids (ball rows untouched) and statistics."""
    people = tracking[tracking.cls != "ball"]
    p = pieces(people)
    max_gap = int(round(max_gap_s * fps))
    cand = []
    for a, ra in p.iterrows():
        nxt = p[(p.start > ra.end) & (p.start <= ra.end + max_gap) & (p.team == ra.team)
                & ((p.cls == "referee") == (ra.cls == "referee"))]
        if nxt.empty:
            continue
        gap_s = (nxt.start - ra.end) / fps
        dist = np.hypot(nxt.x0 - ra.x1, nxt.y0 - ra.y1)
        ok = dist <= MAX_SPEED * gap_s + MARGIN_M
        cand += [(d + MAX_SPEED * g, a, b) for b, d, g in zip(nxt.index[ok], dist[ok], gap_s[ok])]
    cand.sort()

    successor, predecessor = {}, {}
    for _, a, b in cand:
        if a in successor or b in predecessor:
            continue
        successor[a], predecessor[b] = b, a

    # follow each chain from its first piece and give the whole chain that piece's id
    new_id = {}
    for start in p.index:
        if start in predecessor:
            continue
        node = start
        while True:
            new_id[node] = start
            if node not in successor:
                break
            node = successor[node]
    out = tracking.copy()
    is_person = out.cls != "ball"
    out.loc[is_person, "track_id"] = out.loc[is_person, "track_id"].map(new_id)
    stats = {"max_gap_s": max_gap_s, "pieces_before": int(len(p)), "joins": len(successor),
             "pieces_after": int(len(p) - len(successor))}
    return out, stats
