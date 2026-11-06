import numpy as np
import pandas as pd

from touchline.metric_colours import STYLES, classify_metric, colour_table
from touchline.players import ROLES


def fixture(key, value, role=ROLES[2]):
    p = {
        "role": role,
        "rating_profile": role,
        "minutes": 900.0,
        "penalties_taken": 0,
        "penalties_faced": 0,
        key: value,
    }
    peers = pd.DataFrame(
        [{"rating_profile": role, "minutes": 900.0, key: v} for v in [20.0, 40.0, 60.0, 80.0]]
    )
    return p, peers


def test_strength_and_adverse_metrics_have_opposite_directions():
    p, peers = fixture("recoveries", 100)
    assert classify_metric(p, "recoveries", peers)[0] == "green"
    p, peers = fixture("fouls", 100)
    assert classify_metric(p, "fouls", peers)[0] == "yellow"
    p["fouls"] = 0
    assert classify_metric(p, "fouls", peers)[0] == "green"


def test_context_and_goalkeeper_difficulty_are_not_forced_errors():
    p, peers = fixture("goals_conceded", 36, ROLES[0])
    assert classify_metric(p, "goals_conceded", peers)[0] == "blue"
    p, peers = fixture("low_xg_conceded", 11, ROLES[0])
    signals = pd.DataFrame([{"metric": "low_xg_conceded", "colour": "red", "fact": "invalid inference"}])
    colour, reason = classify_metric(p, "low_xg_conceded", peers, signals)
    assert colour == "yellow" and "не доказывает" in reason


def test_not_applicable_missing_and_short_samples_have_neutral_colours():
    p, peers = fixture("save_pct", 74)
    assert classify_metric(p, "save_pct", peers)[0] == "grey"
    p, peers = fixture("recoveries", np.nan)
    assert classify_metric(p, "recoveries", peers)[0] == "grey"
    p["recoveries"], p["minutes"] = 100, 90
    assert classify_metric(p, "recoveries", peers)[0] == "grey"


def test_no_penalty_attempt_is_not_a_perfect_record():
    p, peers = fixture("penalties_missed", 0)
    assert classify_metric(p, "penalties_missed", peers)[0] == "blue"
    p["penalties_taken"], p["penalties_missed"] = 6, 2
    assert classify_metric(p, "penalties_missed", peers)[0] == "red"


def test_existing_evidence_priority_is_preserved():
    p, peers = fixture("provider_errors", 1)
    signals = pd.DataFrame(
        [{"metric": "provider_errors", "colour": "red", "fact": "Error перед голевым ударом"}]
    )
    assert classify_metric(p, "provider_errors", peers, signals)[0] == "red"


def test_every_displayed_row_has_a_colour_and_reason():
    p, peers = fixture("recoveries", 100)
    p.update(penalties_conceded=1, arbitrary_context=10)
    table = colour_table(p, ["recoveries", "penalties_conceded", "arbitrary_context"], peers)
    assert len(table) == 3
    assert set(table.colour).issubset(STYLES)
    assert table.reason.str.len().gt(0).all()


def test_rating_colour_compares_raw_rating_not_rating_per90():
    p, peers = fixture("average_rating", 7.0)
    peers["average_rating"] = [5.0, 5.5, 6.0, 6.5]
    peers["minutes"] = [900.0, 1200.0, 1800.0, 3000.0]
    colour, reason = classify_metric(p, "average_rating", peers)
    assert colour == "green"
    assert "7.00/10" in reason and "/90" not in reason
