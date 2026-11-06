import json

import numpy as np
import pandas as pd
import pytest

from touchline.ingest import flatten
from touchline.players import ROLES
from touchline.season import WEIGHTS, aggregate_season, build_match, rate_matches, set_piece_shot


def event(index, kind, second=0, team=1, player=10, possession=1, **fields):
    return {
        "id": f"s{index}",
        "index": index,
        "period": 1,
        "minute": 0,
        "second": second,
        "timestamp": f"00:00:{second:06.3f}",
        "type": {"name": kind},
        "team": {"id": team, "name": f"T{team}"},
        "player": {"id": player, "name": f"P{player}"},
        "position": {"name": "Center Back"},
        "possession": possession,
        "possession_team": {"id": team},
        "location": [12, 40],
        **fields,
    }


def test_penalty_link_stops_at_unrelated_play_and_handles_paired_foul():
    foul = event(1, "Foul Committed", foul_committed={"penalty": True})
    shot = event(
        3,
        "Shot",
        30,
        2,
        20,
        shot={"type": {"name": "Penalty"}, "outcome": {"name": "Goal"}, "statsbomb_xg": 0.76},
    )
    pair = event(2, "Foul Won", 0, 2, 20, foul_won={"penalty": True})
    assert set_piece_shot([foul, pair, shot], 0, True) == shot
    carry = event(2, "Carry", 5, 2, 20)
    assert set_piece_shot([foul, carry, shot], 0, True) is None


def test_free_kick_link_does_not_cross_new_possession():
    foul = event(1, "Foul Committed")
    kick = event(2, "Pass", 10, 2, 20, 2, **{"pass": {"type": {"name": "Free Kick"}}})
    shot = event(3, "Shot", 20, 2, 20, 2, shot={"type": {"name": "Open Play"}})
    assert set_piece_shot([foul, kick, shot], 0) == shot
    shot["possession"] = 3
    assert set_piece_shot([foul, kick, shot], 0) is None


def test_penalty_red_card_and_goal_have_one_episode_deduction():
    raw = [
        event(
            1,
            "Starting XI",
            tactics={"lineup": [{"player": {"id": 10, "name": "P10"}, "position": {"name": "Center Back"}}]},
        ),
        event(2, "Foul Committed", 10, foul_committed={"penalty": True, "card": {"name": "Red Card"}}),
        event(3, "Foul Won", 10, 2, 20, foul_won={"penalty": True}),
        event(
            4,
            "Shot",
            30,
            2,
            20,
            shot={"type": {"name": "Penalty"}, "statsbomb_xg": 0.76, "outcome": {"name": "Goal"}},
        ),
        event(5, "Half End", 59),
    ]
    rows, shot_rows = flatten(1, raw)
    columns = [
        "event_id",
        "match_id",
        "event_index",
        "period",
        "minute",
        "second",
        "clock",
        "type",
        "team_id",
        "team",
        "player_id",
        "player",
        "possession",
        "possession_team_id",
        "x",
        "y",
        "end_x",
        "end_y",
        "outcome",
        "complete",
        "recipient_id",
        "recipient",
        "under_pressure",
        "card",
        "play_pattern",
    ]
    shot_columns = [
        "event_id",
        "match_id",
        "team_id",
        "player",
        "minute",
        "period",
        "clock",
        "x",
        "y",
        "distance",
        "angle",
        "body_part",
        "technique",
        "shot_type",
        "under_pressure",
        "first_time",
        "goal",
        "statsbomb_xg",
    ]
    frame, _ = build_match(
        raw,
        pd.DataFrame(rows, columns=columns),
        pd.DataFrame(shot_rows, columns=shot_columns),
        {
            "match_id": 1,
            "date": "2015-01-01",
            "home_id": 1,
            "home": "T1",
            "away": "T2",
            "home_score": 0,
            "away_score": 1,
        },
    )
    p = frame[frame.player_id.eq(10)].iloc[0]
    assert p.penalties_conceded == 1 and p.red_cards == 1 and p.fouls_to_goals == 1
    assert p.deduction == pytest.approx(0.8)
    assert len(json.loads(p.deduction_detail)) == 1


def test_rating_weights_and_short_appearances():
    assert all(sum(weights.values()) == pytest.approx(1) for weights in WEIGHTS.values())
    records = []
    for i in range(7):
        record = dict(
            player_id=1,
            minutes=90 if i < 6 else 5,
            role=ROLES[2],
            position="Center Midfield",
            deduction=0.0,
            penalties_saved=0,
            penalties_won=0,
        )
        record.update({feature: 2.0 for feature in WEIGHTS[ROLES[2]]})
        records.append(record)
    frame, reference = rate_matches(pd.DataFrame(records))
    assert frame.rating.iloc[:6].to_numpy() == pytest.approx([6] * 6)
    assert pd.isna(frame.rating.iloc[-1])
    assert reference[ROLES[2]]["recoveries"]["rows"] == 6


def test_season_average_is_minute_weighted_and_ratios_use_denominators():
    frame = pd.DataFrame(
        [
            dict(
                player_id=1,
                player="A",
                role=ROLES[2],
                rating_profile=ROLES[2],
                minutes=90,
                rating=8.0,
                start=1,
                completed_passes=90,
                passes=100,
                saves=2,
                shots_faced_on_target=2,
            ),
            dict(
                player_id=1,
                player="A",
                role=ROLES[2],
                rating_profile=ROLES[2],
                minutes=30,
                rating=4.0,
                start=0,
                completed_passes=1,
                passes=10,
                saves=0,
                shots_faced_on_target=2,
            ),
            dict(
                player_id=1,
                player="A",
                role=ROLES[2],
                rating_profile=ROLES[2],
                minutes=5,
                rating=np.nan,
                start=0,
                completed_passes=0,
                passes=0,
                saves=0,
                shots_faced_on_target=0,
            ),
        ]
    )
    result = aggregate_season(frame).iloc[0]
    assert result.average_rating == 7
    assert result.rated_minutes == 120 and result.minutes == 125
    assert result.completion_pct == pytest.approx(100 * 91 / 110)
    assert result.save_pct == 50


def test_low_xg_goal_is_a_review_flag_without_automatic_deduction():
    raw = [
        event(
            1,
            "Starting XI",
            tactics={"lineup": [{"player": {"id": 10, "name": "P10"}, "position": {"name": "Goalkeeper"}}]},
        ),
        event(
            2,
            "Shot",
            10,
            2,
            20,
            shot={"type": {"name": "Open Play"}, "statsbomb_xg": 0.05, "outcome": {"name": "Goal"}},
        ),
        event(
            3,
            "Goal Keeper",
            10,
            position={"name": "Goalkeeper"},
            goalkeeper={"type": {"name": "Goal Conceded"}},
        ),
        event(4, "Half End", 59),
    ]
    from touchline.ingest import flatten

    rows, shot_rows = flatten(1, raw)
    # Use the production column order without depending on real-data availability.
    import sqlite3

    from touchline.config import ROOT

    with sqlite3.connect(":memory:") as connection:
        connection.executescript((ROOT / "sql/schema.sql").read_text())
        event_columns = [row[1] for row in connection.execute("PRAGMA table_info(events)")]
        shot_columns = [row[1] for row in connection.execute("PRAGMA table_info(shots)")]
    frame, _ = build_match(
        raw,
        pd.DataFrame(rows, columns=event_columns),
        pd.DataFrame(shot_rows, columns=shot_columns),
        {
            "match_id": 1,
            "date": "2015-01-01",
            "home_id": 1,
            "home": "T1",
            "away": "T2",
            "home_score": 0,
            "away_score": 1,
        },
    )
    keeper = frame[frame.player_id.eq(10)].iloc[0]
    assert keeper.low_xg_conceded == 1 and keeper.goals_conceded == 1
    assert keeper.deduction == 0
