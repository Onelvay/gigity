"""Formation facts and comparisons must come from event annotations."""

import base64

import pytest

from touchline.config import DB
from touchline.tactics import ROLE_KEYS, formation_figure, role_family


@pytest.mark.parametrize(
    "position,expected",
    [
        ("Right Center Back", "Центральная защита"),
        ("Left Wing Back", "Крайняя защита"),
        ("Right Defensive Midfield", "Опорная полузащита"),
        ("Center Attacking Midfield", "Атакующая полузащита"),
        ("Right Midfield", "Фланговая полузащита"),
        ("Left Center Midfield", "Центральная полузащита"),
        ("Right Wing", "Вингер"),
        ("Secondary Striker", "Нападение"),
    ],
)
def test_role_comparison_groups(position, expected):
    assert role_family(position) == expected


@pytest.mark.skipif(not DB.exists(), reason="Requires the downloaded season")
def test_real_formations_and_role_filtered_totals():
    from touchline.data import query
    from touchline.tactics import build_tactics, role_comparison

    tactics = build_tactics(query("SELECT * FROM matches"))
    assert len(tactics["lineups"]) == 38
    assert tactics["formations"] == {442: 33, 4141: 1, 4411: 2, 451: 1, 4231: 1}
    assert tactics["common_xi_matches"] == 13
    assert tactics["representative"] == 3754250
    counter = tactics["patterns"]["From Counter"]
    assert counter["shots"] == 38 and counter["goals"] == 6
    assert counter["xg"] == pytest.approx(6.657187175)
    for data in tactics["lineups"].values():
        assert len({p["player_id"] for p in data["lineup"]}) == 11
        assert all(0 < p["x"] < 100 and 0 < p["y"] < 150 for p in data["lineup"])
    lineup = tactics["lineups"][3754250]["lineup"]
    kante = next(p for p in lineup if p["player_id"] == 3961)
    assert kante["number"] == 14 and kante["position"] == "Left Defensive Midfield"
    figure = formation_figure(lineup, 3961)
    assert len(figure.data[0].customdata) == 11
    assert len(figure.layout.images) == 12
    pitch = base64.b64decode(figure.layout.images[0].source.split(",", 1)[1])
    assert b"<svg" in pitch and b"<script" not in pitch
    matches = query("SELECT * FROM player_matches")
    p, colours, family = role_comparison(3961, kante["position"], matches)
    expected = matches[matches.player_id.eq(3961) & matches.position.map(role_family).eq(family)]
    assert p.minutes == pytest.approx(expected.minutes.sum())
    assert p.recoveries == expected.recoveries.sum()
    assert p.matches < 37  # Does not silently substitute whole-season totals.
    assert colours.metric.tolist() == ROLE_KEYS[family]
    assert colours.loc[colours.metric.eq("recoveries"), "colour"].item() == "green"
