"""Transparent descriptive analyses; no causal or pre-match claims."""

import numpy as np
import pandas as pd


def fixed_shot_outcomes(home_xg, away_xg):
    """Exact Poisson-binomial replay under independent, fixed shot probabilities."""

    def distribution(values):
        mass = np.array([1.0])
        for p in values:
            if not 0 <= p <= 1:
                raise ValueError("Shot probabilities must lie in [0,1]")
            mass = np.convolve(mass, [1 - p, p])
        return mass

    home, away = distribution(home_xg), distribution(away_xg)
    joint = np.outer(home, away)
    return {
        "home_win": float(np.tril(joint, -1).sum()),
        "draw": float(np.trace(joint)),
        "away_win": float(np.triu(joint, 1).sum()),
    }


def score_states(events, home_id, away_id):
    """State before each event; own goals counted once using Own Goal Against."""
    frame = events.sort_values("event_index").copy()
    home, away = 0, 0
    states, goals = [], []
    for event in frame.itertuples():
        difference = home - away if event.team_id == home_id else away - home
        states.append("Ведёт" if difference > 0 else "Уступает" if difference < 0 else "Равный счёт")
        scorer = None
        if event.type == "Shot" and event.outcome == "Goal":
            scorer = event.team_id
        elif event.type == "Own Goal Against":
            scorer = away_id if event.team_id == home_id else home_id
        if scorer is not None:
            home += int(scorer == home_id)
            away += int(scorer == away_id)
            goals.append(
                {
                    "event_id": event.event_id,
                    "clock": event.clock,
                    "minute": event.minute,
                    "player": event.player,
                    "scoring_team_id": scorer,
                    "own_goal": event.type == "Own Goal Against",
                    "score": f"{home}–{away}",
                }
            )
    frame["score_state"] = states
    return frame, pd.DataFrame(
        goals, columns=["event_id", "clock", "minute", "player", "scoring_team_id", "own_goal", "score"]
    )


def state_summary(events, team_id, home_id, away_id):
    frame, goals = score_states(events, home_id, away_id)
    minutes = {"Ведёт": 0.0, "Уступает": 0.0, "Равный счёт": 0.0}
    balance, start = 0, 0.0
    for goal in goals.itertuples():
        state = "Ведёт" if balance > 0 else "Уступает" if balance < 0 else "Равный счёт"
        minutes[state] += max(0, goal.clock - start)
        balance += 1 if goal.scoring_team_id == team_id else -1
        start = goal.clock
    state = "Ведёт" if balance > 0 else "Уступает" if balance < 0 else "Равный счёт"
    minutes[state] += max(0, frame.clock.max() - start)
    selected = frame[frame.team_id.eq(team_id)]
    result = []
    for state, duration in minutes.items():
        subset = selected[selected.score_state.eq(state)]
        shots = int(subset.type.eq("Shot").sum())
        result.append(
            {
                "Состояние": state,
                "Минуты": round(duration, 1),
                "Удары": shots,
                "Удары / 90": round(shots / duration * 90, 2) if duration > 0 else None,
                "Давление": int(subset.type.eq("Pressure").sum()),
            }
        )
    return pd.DataFrame(result)
