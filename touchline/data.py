import sqlite3
from contextlib import closing

import pandas as pd

from touchline.config import DB


def query(sql, params=(), db=DB):
    # Open read-only: a missing database should not silently create an empty file.
    with closing(sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)) as connection:
        return pd.read_sql_query(sql, connection, params=params)


def matches():
    return query("SELECT * FROM matches ORDER BY date, kick_off, match_id")


def match_events(match_id):
    return query("SELECT * FROM events WHERE match_id=? ORDER BY event_index", (int(match_id),))


def match_shots(match_id):
    return query("SELECT * FROM shots WHERE match_id=? ORDER BY period, clock", (int(match_id),))


def team_matches():
    return query("SELECT * FROM team_match ORDER BY date, match_id, is_home DESC")


def factor_frame():
    df = team_matches()
    other = df[
        [
            "match_id",
            "team_id",
            "shots",
            "xg",
            "box_entries",
            "pressures",
            "red_cards",
            "completed_passes",
            "passes",
        ]
    ].rename(columns={"team_id": "opponent_id"})
    df = df.merge(other, on=["match_id", "opponent_id"], suffixes=("", "_opp"), validate="one_to_one")
    for name in ("shots", "xg", "box_entries", "pressures", "red_cards"):
        df[f"{name}_diff"] = df[name] - df[f"{name}_opp"]
    df["completion_diff"] = df.completed_passes / df.passes.clip(
        lower=1
    ) - df.completed_passes_opp / df.passes_opp.clip(lower=1)
    return df
