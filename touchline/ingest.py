"""Flatten cached events into an atomic, indexed SQLite research database."""

import json
import math
import sqlite3
from contextlib import closing

from touchline.config import ARTIFACTS, COMPETITION, DB, RAW, ROOT, SEASON, directories
from touchline.download import read_raw


def geometry(x, y):
    """Distance/angle on StatsBomb's 120 x 80 grid; goalposts at (120,36/44)."""
    dx, dy = 120 - x, 40 - y
    distance = math.hypot(dx, dy)
    # atan2 cross/dot is stable on the goal line, unlike atan(8*dx/...).
    angle = math.atan2(abs(8 * dx), dx * dx + (36 - y) * (44 - y))
    return distance, angle


def flatten(match_id, events):
    rows, shots = [], []
    lengths = {}
    for e in events:
        h, m, s = map(float, e["timestamp"].split(":"))
        lengths[e["period"]] = max(lengths.get(e["period"], 0), h * 60 + m + s / 60)
    offsets, running = {}, 0
    for period in sorted(lengths):
        offsets[period] = running
        running += max(lengths[period], 45 if period <= 2 else 15)
    for e in events:
        kind = e["type"]["name"]
        payload = e.get(kind.lower().replace(" ", "_"), {})
        start = e.get("location", [None, None])
        end = payload.get("end_location", [None, None])
        h, m, s = map(float, e["timestamp"].split(":"))
        clock = offsets[e["period"]] + h * 60 + m + s / 60
        team, player = e.get("team", {}), e.get("player", {})
        card = payload.get("card", {}).get("name")
        outcome = payload.get("outcome", {}).get("name")
        complete = int(kind == "Carry" or (kind == "Pass" and outcome is None))
        recipient = payload.get("recipient", {})
        rows.append(
            (
                e["id"],
                match_id,
                e["index"],
                e["period"],
                e["minute"],
                e["second"],
                clock,
                kind,
                team.get("id"),
                team.get("name"),
                player.get("id"),
                player.get("name"),
                e.get("possession"),
                e.get("possession_team", {}).get("id"),
                *start[:2],
                *end[:2],
                outcome,
                complete,
                recipient.get("id"),
                recipient.get("name"),
                int(e.get("under_pressure", False)),
                card,
                e.get("play_pattern", {}).get("name"),
            )
        )
        if kind == "Shot":
            if start[0] is None or start[1] is None or "statsbomb_xg" not in payload:
                raise ValueError(f"Shot {e['id']} lacks coordinates or provider xG")
            distance, angle = geometry(*start[:2])
            shots.append(
                (
                    e["id"],
                    match_id,
                    team["id"],
                    player.get("name"),
                    e["minute"],
                    e["period"],
                    clock,
                    *start[:2],
                    distance,
                    angle,
                    payload.get("body_part", {}).get("name", "Unknown"),
                    payload.get("technique", {}).get("name", "Unknown"),
                    payload.get("type", {}).get("name", "Unknown"),
                    int(e.get("under_pressure", False)),
                    int(payload.get("first_time", False)),
                    int(outcome == "Goal"),
                    payload["statsbomb_xg"],
                )
            )
    return rows, shots


def main():
    directories()
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    matches = read_raw(f"matches/{COMPETITION}/{SEASON}.json")
    available = [m for m in matches if (RAW / f"events/{m['match_id']}.json.gz").exists()]
    if not available:
        raise SystemExit("Run python -m touchline.download first")
    temporary = DB.with_suffix(".building.sqlite")
    if temporary.exists():
        temporary.unlink()
    with closing(sqlite3.connect(temporary)) as conn:
        conn.executescript((ROOT / "sql/schema.sql").read_text(encoding="utf-8"))
        for i, m in enumerate(available, 1):
            conn.execute(
                "INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    m["match_id"],
                    m["match_date"],
                    m["kick_off"],
                    m["home_team"]["home_team_id"],
                    m["home_team"]["home_team_name"],
                    m["away_team"]["away_team_id"],
                    m["away_team"]["away_team_name"],
                    m["home_score"],
                    m["away_score"],
                    m.get("match_week"),
                    m.get("stadium", {}).get("name"),
                    manifest["commit"],
                ),
            )
            rows, shots = flatten(m["match_id"], read_raw(f"events/{m['match_id']}.json"))
            conn.executemany("INSERT INTO events VALUES (" + ",".join(["?"] * 25) + ")", rows)
            conn.executemany("INSERT INTO shots VALUES (" + ",".join(["?"] * 18) + ")", shots)
            if i % 50 == 0 or i == len(available):
                conn.commit()
                print(f"Ingested {i}/{len(available)}", flush=True)
        conn.execute("ANALYZE")
        conn.commit()
    temporary.replace(DB)
    print(f"Database ready: {len(available)} matches")


if __name__ == "__main__":
    main()
