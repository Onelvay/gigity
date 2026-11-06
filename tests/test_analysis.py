import gzip
import json
import math
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from touchline.analytics import fixed_shot_outcomes, score_states, state_summary
from touchline.config import ROOT
from touchline.evidence import deterministic_report, validate_report
from touchline.ingest import flatten, geometry
from touchline.modeling import FACTORS, SHOT_FEATURES, explain_factors, masks, split_dates


def event(index, kind, team=1, minute=0, period=1, **payload):
    return {
        "id": f"event-{index}",
        "index": index,
        "type": {"name": kind},
        "period": period,
        "timestamp": f"00:{minute:02d}:00.000",
        "minute": minute + (45 if period == 2 else 0),
        "second": 0,
        "team": {"id": team, "name": f"Team {team}"},
        "player": {"id": team * 10, "name": f"Player {team}"},
        "location": [110, 40],
        **payload,
    }


def test_geometry_uses_goal_angle_without_division_by_zero():
    distance, angle = geometry(110, 40)
    assert distance == 10
    assert angle == pytest.approx(2 * math.atan(4 / 10))
    assert geometry(120, 40)[1] == pytest.approx(math.pi)
    assert geometry(110, 30)[1] == pytest.approx(geometry(110, 50)[1])


def test_fixed_shot_replay_known_distribution_and_no_shots():
    odds = fixed_shot_outcomes([0.5], [0.5])
    assert odds == {"home_win": 0.25, "draw": 0.5, "away_win": 0.25}
    assert fixed_shot_outcomes([], []) == {"home_win": 0.0, "draw": 1.0, "away_win": 0.0}
    assert sum(fixed_shot_outcomes([0.1, 0.4, 0.9], [0.3, 0.8]).values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        fixed_shot_outcomes([1.2], [])


def test_ingestion_goal_own_goal_and_box_entry_are_consistent():
    events = [
        event(1, "Pass", pass_={}),
        event(2, "Shot", minute=5, shot={"outcome": {"name": "Goal"}, "statsbomb_xg": 0.25}),
        event(3, "Own Goal Against", team=1, minute=10),
        event(4, "Own Goal For", team=2, minute=10),
        event(5, "Carry", team=2, minute=12, carry={"end_location": [110, 40]}, location=[90, 40]),
        event(6, "Half End", minute=47),
        event(7, "Half Start", period=2),
        event(8, "Half End", period=2, minute=46),
    ]
    rows, shots = flatten(100, events)
    with sqlite3.connect(":memory:") as connection:
        connection.executescript((ROOT / "sql/schema.sql").read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO matches VALUES (100,'2015-01-01','12:00',1,'Team 1',2,'Team 2',1,1,1,'Stadium','abc')"
        )
        connection.executemany("INSERT INTO events VALUES (" + ",".join(["?"] * 25) + ")", rows)
        connection.executemany("INSERT INTO shots VALUES (" + ",".join(["?"] * 18) + ")", shots)
        table = pd.read_sql_query("SELECT * FROM team_match ORDER BY team_id", connection)
        assert table.xg.tolist() == [0.25, 0]
        assert table.result.tolist() == ["D", "D"]
        assert table.box_entries.tolist() == [0, 1]
        df = pd.read_sql_query("SELECT * FROM events", connection)
    states, goals = score_states(df, 1, 2)
    assert goals.score.tolist() == ["1–0", "1–1"]  # No duplication from Own Goal For.
    assert states.loc[states.event_id.eq("event-2"), "score_state"].item() == "Равный счёт"
    assert states.loc[states.event_id.eq("event-3"), "score_state"].item() == "Ведёт"
    assert rows[-1][6] == pytest.approx(93)  # Includes first-half stoppage time.
    summary = state_summary(df, 1, 1, 2)
    assert summary["Минуты"].sum() == pytest.approx(93)


def test_chronological_split_keeps_match_and_same_day_together():
    frame = pd.DataFrame(
        {"match_id": range(150), "date": pd.date_range("2015-01-01", periods=150).strftime("%Y-%m-%d")}
    )
    duplicated = pd.concat([frame, frame], ignore_index=True)
    cutoffs = split_dates(frame)
    parts = masks(duplicated, cutoffs)
    ids = [set(duplicated.loc[mask, "match_id"]) for mask in parts]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert all(mask.sum() % 2 == 0 for mask in parts)
    assert duplicated.loc[parts[0], "date"].max() < duplicated.loc[parts[1], "date"].min()


def test_model_inputs_exclude_labels_and_post_shot_information():
    forbidden = {"goal", "statsbomb_xg", "outcome", "end_x", "end_y", "home_score", "away_score"}
    assert not forbidden.intersection(SHOT_FEATURES)
    assert not {"goals", "conceded", "points", "result"}.intersection(FACTORS)


def test_linear_explanation_reconstructs_probability_contrast():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(120, len(FACTORS))), columns=FACTORS)
    y = np.array(["W", "D", "L"] * 40)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    explanation = explain_factors(model, x.iloc[0])
    probabilities = explanation["probabilities"]
    log_ratio = math.log(probabilities[explanation["predicted"]] / probabilities[explanation["reference"]])
    assert sum(explanation["contributions"].values()) + explanation["intercept"] == pytest.approx(log_ratio)


def test_report_rejects_fabricated_and_missing_references():
    data = {
        "evidence": [
            {"id": "result", "fact": "A 1–0 B"},
            {"id": "team_1_xg", "fact": "A xG = 1.2"},
            {"id": "team_1_box_entries", "fact": "A entries = 5"},
        ]
    }
    report = deterministic_report(data)
    assert validate_report(report, data) == report
    report["sections"][0]["evidence_ids"] = ["invented-event"]
    with pytest.raises(ValueError):
        validate_report(report, data)
    report["sections"][0]["evidence_ids"] = []
    with pytest.raises(ValueError):
        validate_report(report, data)


def test_downloader_detects_corrupted_cached_source(tmp_path, monkeypatch):
    from touchline import download

    monkeypatch.setattr(download, "RAW", tmp_path)
    (tmp_path / "competitions.json.gz").write_bytes(gzip.compress(b"[]"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        download.download_one("pinned-revision", "competitions.json", "incorrect-hash")


def test_llm_request_uses_schema_and_validated_evidence(monkeypatch):
    import openai

    from touchline.evidence import llm_report

    data = {"evidence": [{"id": "result", "fact": "A 1–0 B"}]}
    answer = {
        "title": "Разбор",
        "sections": [{"heading": "Результат", "interpretation": "A выиграла.", "evidence_ids": ["result"]}],
        "limitations": ["Наблюдательные данные"],
    }
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text=json.dumps(answer))

    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(
        openai, "OpenAI", lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(create=create))
    )
    assert llm_report(data) == answer
    assert captured["store"] is False
    assert captured["text"]["format"]["strict"] is True
    assert json.loads(captured["input"]) == data
    assert llm_report({"scope": "player-season", **data}) == answer
    assert "сезон указанного игрока" in captured["instructions"]
