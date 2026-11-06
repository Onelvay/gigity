import pandas as pd
import pytest

from touchline.ingest import flatten
from touchline.players import ROLES, analyze_players, find_incidents, player_roster, role


def e(index, kind, team=1, second=0, period=1, player=10, possession=1, **fields):
    return {
        "id": f"e{index}",
        "index": index,
        "type": {"name": kind},
        "period": period,
        "timestamp": f"00:00:{second:06.3f}",
        "minute": 0 if period == 1 else 45,
        "second": int(second),
        "team": {"id": team, "name": f"T{team}"},
        "player": {"id": player, "name": f"P{player}"},
        "position": {"name": "Center Midfield"},
        "possession": possession,
        "possession_team": {"id": team},
        "location": [50, 40],
        **fields,
    }


def frames(raw):
    rows, shots = flatten(1, raw)
    events = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
    shots = pd.DataFrame(
        shots,
        columns=[
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
        ],
    )
    return events, shots


def shot(index=3, second=10, period=1, possession=2):
    return e(
        index,
        "Shot",
        team=2,
        player=20,
        second=second,
        period=period,
        possession=possession,
        shot={"statsbomb_xg": 0.4, "outcome": {"name": "Goal"}},
    )


def test_roles_are_specific_to_football_positions():
    assert role("Goalkeeper") == ROLES[0]
    assert role("Left Wing Back") == ROLES[1]
    assert role("Center Attacking Midfield") == ROLES[2]
    assert role("Right Wing") == ROLES[3]
    assert role("Center Forward") == ROLES[3]
    assert role("Unknown") == "Позиция не определена"


def test_lost_control_is_observed_but_not_automatically_an_error():
    raw = [
        e(1, "Miscontrol", second=1, under_pressure=True),
        e(2, "Carry", team=2, player=20, second=3, possession=2, carry={"end_location": [70, 40]}),
        shot(),
    ]
    events, _ = frames(raw)
    found = find_incidents(raw, events, player_roster(raw, events))
    assert len(found) == 1
    action = found.iloc[0]
    assert action.kind == "Неудачная обработка"
    assert not action.provider_error
    assert action.shot_id == "e3" and action.shot_goal
    assert action.seconds_after == 9
    assert action.shot_xg == 0.4
    assert action.under_pressure == 1


@pytest.mark.parametrize(
    "break_event",
    [
        e(2, "Pass", second=2, pass_={}, possession=3),
        e(2, "Foul Committed", team=2, player=20, second=2),
        e(
            2,
            "Pass",
            team=2,
            player=20,
            second=2,
            possession=2,
            **{"pass": {"type": {"name": "Free Kick"}, "end_location": [60, 40]}},
        ),
    ],
)
def test_no_link_across_regained_control_foul_or_restart(break_event):
    raw = [e(1, "Miscontrol", second=1), break_event, shot()]
    events, _ = frames(raw)
    found = find_incidents(raw, events, player_roster(raw, events))
    assert found.iloc[0].shot_id is None


def test_no_link_across_half_or_time_limit():
    for later_shot in [shot(second=30), shot(second=5, period=2)]:
        raw = [e(1, "Miscontrol", second=1), later_shot]
        events, _ = frames(raw)
        found = find_incidents(raw, events, player_roster(raw, events), window=20)
        assert found.iloc[0].shot_id is None


def test_same_shot_only_counted_once_for_player_totals():
    raw = [e(1, "Miscontrol", second=1), e(2, "Dribbled Past", player=11, second=2), shot()]
    events, _ = frames(raw)
    found = find_incidents(raw, events, player_roster(raw, events))
    assert found.shot_id.notna().sum() == 2
    assert found.primary_link.sum() == 1
    assert found.loc[found.primary_link, "player_id"].item() == 11


def test_pass_out_and_offside_do_not_generate_counterattack_claims():
    for outcome in ["Out", "Pass Offside"]:
        raw = [e(1, "Pass", second=1, **{"pass": {"outcome": {"name": outcome}}}), shot()]
        events, _ = frames(raw)
        found = find_incidents(raw, events, player_roster(raw, events))
        assert not found.iloc[0].provider_error
        assert found.iloc[0].shot_id is None


def test_key_pass_xa_and_substitution_minutes_use_source_events():
    lineup = [{"player": {"id": 10, "name": "P10"}, "position": {"name": "Center Midfield"}}]
    raw = [
        e(1, "Starting XI", second=0, tactics={"lineup": lineup}),
        e(
            2,
            "Pass",
            second=1,
            **{
                "pass": {
                    "assisted_shot_id": "e3",
                    "shot_assist": True,
                    "goal_assist": True,
                    "end_location": [110, 40],
                }
            },
        ),
        e(3, "Shot", second=2, player=11, shot={"statsbomb_xg": 0.3, "outcome": {"name": "Goal"}}),
        e(4, "Substitution", second=30, substitution={"replacement": {"id": 12, "name": "P12"}}),
        e(5, "Carry", second=40, player=12, carry={"end_location": [60, 40]}),
        e(6, "Half End", second=59, player=12),
    ]
    events, shots = frames(raw)
    players, _ = analyze_players(raw, events, shots)
    source = players[players.player_id.eq(10)].iloc[0]
    assert source.key_passes == 1 and source.assists == 1
    assert source.xa == 0.3 and source.progressive_passes == 1
    assert source.minutes == 0.5
    assert players.loc[players.player_id.eq(12), "minutes"].item() == 0.5
    # Unplayed alternatives are never inferred from a lack of key passes.
    assert "missed_key_passes" not in players.columns
