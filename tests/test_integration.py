"""Real-data checks are skipped in CI until the source dataset is downloaded."""

import json

import pytest

from touchline.config import ARTIFACTS, DB, ROOT

pytestmark = pytest.mark.skipif(not DB.exists(), reason="Download and ingest StatsBomb data first")


def test_real_dataset_scores_and_ranges():
    from touchline.research import audit

    result = audit()
    assert result["matches"] > 0
    assert result["score_mismatches"] == 0
    assert result["invalid_shot_xg"] == 0
    assert result["gross_coordinate_errors"] == 0
    assert result["excluded_own_xg_shots"] == result["invalid_shot_coordinates"]


def test_every_match_has_two_opposing_team_rows():
    from touchline.data import factor_frame

    frame = factor_frame()
    assert frame.groupby("match_id").size().eq(2).all()
    assert frame.groupby("match_id").xg_diff.sum().abs().max() < 1e-8
    assert not frame.isna().any().any()


@pytest.mark.skipif(not (ARTIFACTS / "season_meta.json").exists(), reason="Build the Leicester season first")
def test_application_pages_render_without_exceptions():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    assert not app.exception
    assert next(w for w in app.radio if w.label == "Раздел").value == "Менеджеру"
    next(w for w in app.radio if w.label == "Раздел").set_value("Игрок").run()
    assert next(w for w in app.selectbox if w.label == "Игрок").value == 3961
    for pid in [3815, 40123, 3814, 10960, 6296]:
        next(w for w in app.selectbox if w.label == "Игрок").set_value(pid).run()
        assert not app.exception
    for page in ["Схема и роли", "Состав", "Матчи", "Оценки и данные", "Менеджеру", "Игрок"]:
        next(w for w in app.radio if w.label == "Раздел").set_value(page).run()
        assert not app.exception, f"{page}: {app.exception}"
    next(w for w in app.radio if w.label == "Раздел").set_value("Схема и роли").run()
    from touchline.data import query
    from touchline.tactics import build_tactics

    tactics = build_tactics(query("SELECT * FROM matches"))
    other_match = next(mid for mid, row in tactics["lineups"].items() if row["formation"] == "4231")
    next(w for w in app.selectbox if w.label == "Стартовый состав матча").set_value(other_match).run()
    assert not app.exception
    next(w for w in app.selectbox if w.label == "Игрок на схеме").set_value(3815).run()
    assert not app.exception


@pytest.mark.skipif(not (ARTIFACTS / "season_meta.json").exists(), reason="Build the Leicester season first")
def test_leicester_season_reconciles_with_match_data():
    from touchline.data import query

    players = query("SELECT * FROM leicester_players")
    games = query("SELECT * FROM player_matches WHERE team_id=22")
    assert games.match_id.nunique() == 38
    assert len(players) == 23
    assert games.rating.dropna().between(0, 10).all()
    assert games.loc[games.minutes < 30, "rating"].isna().all()
    keeper = players[players.player_id.eq(3815)].iloc[0]
    assert keeper.minutes == 3420 and keeper.goals_conceded == 36
    assert players.loc[players.player_id.eq(10960), "goals"].item() == 24
    assert players.penalties_conceded.sum() == keeper.penalties_faced
    assert players.penalties_taken.sum() == players.penalties_scored.sum() + players.penalties_missed.sum()


@pytest.mark.skipif(not (ARTIFACTS / "metrics.json").exists(), reason="Train the models first")
def test_artifact_split_has_no_temporal_overlap():
    import pandas as pd

    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    shots = pd.read_csv(ARTIFACTS / "shot_predictions.csv")
    assert shots.groupby("match_id")["split"].nunique().max() == 1
    assert shots.loc[shots.split.eq("train"), "date"].max() <= metrics["cutoffs"]["train_end"]
    assert shots.loc[shots.split.eq("test"), "date"].min() > metrics["cutoffs"]["validation_end"]


def test_real_goalkeeper_distribution_episode_and_conceded_goals():
    from touchline.data import match_events, match_shots
    from touchline.download import read_raw
    from touchline.players import analyze_players

    mid = 3753977  # Watford 1–2 Stoke, source revision pinned in config.
    players, incidents = analyze_players(read_raw(f"events/{mid}.json"), match_events(mid), match_shots(mid))
    action = incidents[incidents.event_id.eq("417dbf67-0319-4f7f-ad1f-5c1ea58769b7")].iloc[0]
    assert action.role == "Вратари"
    assert action.kind == "Неточная передача"
    assert action.shot_goal and action.primary_link
    assert action.seconds_after == 5.4
    assert action.shot_id == "494fed16-e21b-43cd-a4e0-f8ef2e0ecaf0"
    assert not action.provider_error  # Do not invent the provider's Error annotation.
    keeper = players[players.player_id.eq(action.player_id)].iloc[0]
    assert keeper.goals_conceded == 2
    linked = incidents[incidents.primary_link]
    assert not linked.duplicated(["team_id", "shot_id"]).any()
