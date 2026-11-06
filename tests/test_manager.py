import json

import numpy as np
import pandas as pd

from touchline import manager
from touchline.players import ROLES
from touchline.season import WEIGHTS


def person(**overrides):
    fields = {key: 0.0 for weights in WEIGHTS.values() for key in weights}
    fields.update(
        {
            key: 0.0
            for key in [
                "passes",
                "losses_own_half",
                "shots",
                "finishing",
                "dangerous_fouls",
                "yellow_cards",
                "penalties_conceded",
                "red_cards",
                "own_goals",
                "penalties_missed",
                "penalties_scored",
                "penalties_taken",
                "minutes_after_red",
                "handling_failures",
                "provider_errors",
                "low_xg_conceded",
            ]
        }
    )
    fields.update(
        player_id=1,
        name="Player A",
        role=ROLES[2],
        rating_profile=ROLES[2],
        minutes=900.0,
        reliable=1,
        recoveries=100.0,
        completion_pct=np.nan,
        save_pct=np.nan,
    )
    fields.update(overrides)
    return fields


def episode(event_id, **overrides):
    row = dict(
        player_id=1,
        event_id=event_id,
        match_id=100,
        date="2015-01-01",
        minute=20,
        second=0,
        kind="Пенальти в свои ворота",
        shot_id=None,
        provider_error=0,
        primary_link=0,
        goal=0,
        event_x=12.0,
        event_y=40.0,
        event_type="Foul Committed",
        event_card=None,
    )
    row.update(overrides)
    return row


def run(monkeypatch, p, rows=None):
    peers = []
    for i, recoveries in enumerate([20, 40, 60, 80, 100]):
        peer = dict(p, player_id=i + 1, reliable=1, minutes=900.0, recoveries=float(recoveries))
        peers.append(peer)
    monkeypatch.setattr(manager, "aggregate_season", lambda unused: pd.DataFrame(peers))
    episodes = pd.DataFrame(rows or [], columns=list(episode("dummy")))
    return manager.build_manager(pd.DataFrame([p]), pd.DataFrame(), episodes)


def test_strength_and_critical_event_can_coexist(monkeypatch):
    p = person(penalties_conceded=1)
    summary, signals = run(monkeypatch, p, [episode("penalty")])
    assert summary.iloc[0].contribution_colour == "green"
    assert summary.iloc[0].risk_colour == "red"
    assert signals[signals.metric.eq("recoveries")].iloc[0].colour == "green"
    red = signals[signals.metric.eq("penalties_conceded")].iloc[0]
    assert json.loads(red.event_ids) == ["penalty"]
    assert "Разобрать" in red.action


def test_short_sample_is_not_green_even_with_high_rates(monkeypatch):
    summary, signals = run(monkeypatch, person(minutes=90.0, reliable=0, recoveries=500.0))
    assert not signals.colour.eq("green").any()
    assert summary.iloc[0].contribution_colour == "grey"
    assert summary.iloc[0].risk_colour == "grey"


def test_low_xg_keeper_goals_do_not_become_red_errors(monkeypatch):
    p = person(role=ROLES[0], rating_profile=ROLES[0], low_xg_conceded=11)
    summary, signals = run(monkeypatch, p, [episode("shot", kind="Пропущен удар с xG ≤ 0,10", goal=1)])
    row = signals[signals.metric.eq("low_xg_conceded")].iloc[0]
    assert row.colour == "yellow"
    assert "неизвестна" in row.fact and "не считать" in row.action.lower()
    assert summary.iloc[0].risk_colour == "yellow"


def test_same_scoring_shot_is_not_counted_twice(monkeypatch):
    rows = [
        episode("a", kind="Неточная передача", shot_id="one-shot", primary_link=1, goal=1),
        episode("b", kind="Мяч отобрали", shot_id="one-shot", primary_link=1, goal=1),
    ]
    _, signals = run(monkeypatch, person(), rows)
    linked = signals[signals.metric.eq("linked_goals")].iloc[0]
    assert "После действий: 1 " in linked.fact
    assert linked.colour == "yellow"
    assert "не назначать единственного виновника" in linked.action


def test_evidence_never_uses_another_players_episode(monkeypatch):
    _, signals = run(
        monkeypatch, person(penalties_conceded=1), [episode("ours"), episode("other", player_id=2)]
    )
    row = signals[signals.metric.eq("penalties_conceded")].iloc[0]
    assert json.loads(row.event_ids) == ["ours"]


def test_missing_values_and_tied_zero_metrics_are_not_strengths(monkeypatch):
    _, signals = run(monkeypatch, person(recoveries=0.0))
    assert signals[signals.metric.isin(["completion_pct", "xa", "key_passes"])].empty
    assert manager.percentile(np.nan, [1, 2]) is None
    assert manager.percentile(1, []) is None


def test_two_missed_penalties_are_review_priority_not_blame_for_defeat(monkeypatch):
    p = person(penalties_taken=6, penalties_scored=4, penalties_missed=2)
    _, signals = run(
        monkeypatch,
        p,
        [episode("p1", kind="Нереализованный пенальти"), episode("p2", kind="Нереализованный пенальти")],
    )
    row = signals[signals.metric.eq("penalties_missed")].iloc[0]
    assert row.colour == "red" and "4/6" in row.fact
    assert "число попыток невелико" in row.action
