"""Season-wide player review. Leicester-only UI, EPL role peers for descriptive ratings."""

import argparse
import json
import sqlite3
from collections import defaultdict
from contextlib import closing

import numpy as np
import pandas as pd

from touchline.config import ARTIFACTS, DB, REPORTS, directories
from touchline.data import match_events, match_shots, matches, query
from touchline.download import read_raw
from touchline.players import ROLES, analyze_players

TEAM = "Leicester City"
TEAM_ID = 22
VERSION = "leicester-1"
NAMES = {
    3961: "Н’Голо Канте",
    3814: "Рияд Марез",
    10960: "Джейми Варди",
    3815: "Каспер Шмейхель",
    3813: "Уэс Морган",
    40123: "Роберт Хут",
    3812: "Кристиан Фукс",
    3270: "Дэнни Симпсон",
    3662: "Дэнни Дринкуотер",
    3307: "Марк Олбрайтон",
    3810: "Синдзи Окадзаки",
    3671: "Леонардо Ульоа",
    3520: "Джеффри Шлупп",
    3665: "Энди Кинг",
    3312: "Демарай Грей",
}

# Weights sum to one within each role. All counting features are per 90 minutes.
WEIGHTS = {
    ROLES[0]: {"saves": 0.30, "save_pct": 0.30, "completion_pct": 0.15, "collections": 0.15, "sweeps": 0.10},
    ROLES[1]: {
        "tackles_won": 0.25,
        "successful_interceptions": 0.20,
        "blocks": 0.15,
        "clearances": 0.10,
        "recoveries": 0.15,
        "completion_pct": 0.15,
    },
    ROLES[2]: {
        "recoveries": 0.20,
        "tackles_won": 0.10,
        "successful_interceptions": 0.10,
        "progressive_passes": 0.20,
        "key_passes": 0.15,
        "xa": 0.15,
        "completion_pct": 0.10,
    },
    "Атакующая полузащита": {
        "nonpenalty_goals": 0.20,
        "xa": 0.20,
        "assists": 0.15,
        "key_passes": 0.15,
        "progressive_passes": 0.15,
        "dribbles_won": 0.10,
        "completion_pct": 0.05,
    },
    ROLES[3]: {
        "nonpenalty_goals": 0.25,
        "npxg": 0.15,
        "finishing": 0.15,
        "xa": 0.20,
        "key_passes": 0.10,
        "shots_on_target": 0.10,
        "dribbles_won": 0.05,
    },
}
RATIOS = {"completion_pct", "save_pct"}
SUCCESS = {"Won", "Success", "Success In Play", "Success Out"}


def payload(event):
    kind = event["type"]["name"]
    return event.get("goalkeeper" if kind == "Goal Keeper" else kind.lower().replace(" ", "_"), {})


def set_piece_shot(raw, start, penalty=False):
    """First shot in the foul's restart/advantage possession, not any later attack."""
    origin = raw[start]
    origin_time = origin["minute"] * 60 + origin["second"]
    active = bool(payload(origin).get("advantage"))
    possession = None
    for following in raw[start + 1 :]:
        if following["period"] != origin["period"]:
            break
        delta = following["minute"] * 60 + following["second"] - origin_time
        if delta > 180:
            break
        kind, p = following["type"]["name"], payload(following)
        opponent = following["team"]["id"] != origin["team"]["id"]
        if kind == "Foul Won" and delta <= 2:
            continue  # Paired annotation of the same foul.
        if kind in {"Foul Committed", "Foul Won", "Half End", "Offside"}:
            break
        if kind == "Shot" and opponent and p.get("type", {}).get("name") == "Penalty":
            return following if penalty else None
        if penalty:
            if kind in {"Pass", "Carry", "Shot"}:
                break
            continue
        if kind == "Pass" and opponent and p.get("type", {}).get("name") == "Free Kick":
            active = True
            possession = following.get("possession")
        elif kind == "Pass" and p.get("type", {}).get("name") in {
            "Corner",
            "Throw-in",
            "Goal Kick",
            "Kick Off",
        }:
            break
        if kind in {"Pass", "Carry", "Dribble", "Shot"}:
            if not opponent:
                break
            if not active:
                break
            if possession is None:
                possession = following.get("possession")
            if following.get("possession") != possession:
                break
            if kind == "Shot":
                return following
        if following.get("out"):
            break
    return None


def build_match(raw, events, shots, match):
    raw = sorted(raw, key=lambda e: e["index"])
    players, incidents = analyze_players(raw, events, shots)
    event_indices = events.set_index("event_index")

    def regulation_time(index, default):
        if pd.isna(index) or index not in event_indices.index:
            return default
        event = event_indices.loc[index]
        return min(45 if event.period == 1 else 90, event.minute + event.second / 60)

    by_player = defaultdict(list)
    for e in raw:
        if "player" in e:
            by_player[e["player"]["id"]].append(e)
    # One charge per player/episode; the highest applicable cost wins.
    charges = defaultdict(dict)
    incident_rows = []

    def charge(pid, key, value, reason):
        old = charges[pid].get(key, (0, ""))
        if value > old[0]:
            charges[pid][key] = (value, reason)

    for a in incidents.to_dict("records"):
        key = "shot:" + a["shot_id"] if a["shot_id"] else "event:" + a["event_id"]
        base = {
            "Неудачная обработка": 0.04,
            "Мяч отобрали": 0.025,
            "Неточная передача": 0.005,
            "Обыгран соперником": 0.035,
            "Неудачная обводка": 0.02,
        }.get(a["kind"], 0)
        cost = max(base, 0.35 if a["provider_error"] else 0, 0.50 if a["shot_goal"] else 0.15 * a["shot_xg"])
        charge(a["player_id"], key, cost, a["kind"])
        if a["team_id"] == TEAM_ID:
            incident_rows.append(
                {
                    "match_id": match["match_id"],
                    "event_id": a["event_id"],
                    "player_id": a["player_id"],
                    "minute": a["minute"],
                    "second": a["second"],
                    "kind": a["kind"],
                    "shot_id": a["shot_id"],
                    "shot_xg": a["shot_xg"],
                    "goal": int(a["shot_goal"]),
                    "seconds_after": a["seconds_after"],
                    "provider_error": int(a["provider_error"]),
                    "primary_link": int(a["primary_link"]),
                    "note": a["observation"],
                }
            )
    fouls = {}
    for index, e in enumerate(raw):
        if e["type"]["name"] == "Foul Committed":
            fouls[e["id"]] = set_piece_shot(raw, index, payload(e).get("penalty", False))
    output = []
    for record in players.to_dict("records"):
        pid = record["player_id"]
        if pd.notna(record["start_index"]):
            record["minutes"] = round(
                max(
                    0,
                    regulation_time(record["end_index"], 90)
                    - (0 if record["start_index"] == 0 else regulation_time(record["start_index"], 0)),
                ),
                1,
            )
        actions = by_player[pid]
        own_shots = [e for e in actions if e["type"]["name"] == "Shot"]
        penalty_shots = [e for e in own_shots if payload(e).get("type", {}).get("name") == "Penalty"]
        np_shots = [e for e in own_shots if e not in penalty_shots]
        nonpenalty_goals = sum(payload(e)["outcome"]["name"] == "Goal" for e in np_shots)
        npxg = sum(payload(e)["statsbomb_xg"] for e in np_shots)
        won_recoveries = sum(
            e["type"]["name"] == "Ball Recovery" and not payload(e).get("recovery_failure", False)
            for e in actions
        )
        won_interceptions = sum(
            e["type"]["name"] == "Interception" and payload(e).get("outcome", {}).get("name") in SUCCESS
            for e in actions
        )
        fouls_committed = [e for e in actions if e["type"]["name"] == "Foul Committed"]
        dangerous_fouls = [
            e
            for e in fouls_committed
            if e.get("location", [120, 0])[0] <= 30 and 15 <= e.get("location", [120, 0])[1] <= 65
        ]
        p_conceded = sum(payload(e).get("penalty", False) for e in fouls_committed)
        yellow, red, minutes_after_red = 0, 0, 0.0
        for e in actions:
            p, kind = payload(e), e["type"]["name"]
            card = p.get("card", {}).get("name")
            yellow += int(card in {"Yellow Card", "Second Yellow"})
            is_red = card in {"Red Card", "Second Yellow"}
            red += int(is_red)
            if is_red:
                minutes_after_red += max(0, 90 - min(90, e["minute"] + e["second"] / 60))
            shot = fouls.get(e["id"])
            key = "shot:" + shot["id"] if shot else "event:" + e["id"]
            costs = [(0.0, "")]
            if kind == "Foul Committed":
                costs.append(
                    (
                        0.80 if p.get("penalty") else 0.06 if e in dangerous_fouls else 0.015,
                        "Пенальти" if p.get("penalty") else "Фол",
                    )
                )
            if shot:
                costs.append(
                    (
                        0.50
                        if payload(shot)["outcome"]["name"] == "Goal"
                        else 0.15 * payload(shot)["statsbomb_xg"],
                        "После фола: удар",
                    )
                )
            if card:
                costs.append((0.80 if is_red else 0.10, "Удаление" if is_red else "Жёлтая карточка"))
            if kind == "Own Goal Against":
                costs.append((0.70, "Автогол"))
            if e in penalty_shots and p["outcome"]["name"] != "Goal":
                costs.append((0.40, "Нереализованный пенальти"))
            if (
                kind == "Goal Keeper"
                and p.get("type", {}).get("name") in {"Collected", "Punch", "Smother", "Keeper Sweeper"}
                and p.get("outcome", {}).get("name") == "Fail"
            ):
                costs.append((0.35, "Неудачное действие на выходе"))
            value, reason = max(costs, key=lambda pair: pair[0])
            charge(pid, key, value, reason)
            if record["team_id"] == TEAM_ID and (
                kind in {"Foul Committed", "Own Goal Against"}
                or card
                or (e in penalty_shots and p["outcome"]["name"] != "Goal")
                or reason == "Неудачное действие на выходе"
            ):
                label = (
                    "Пенальти в свои ворота"
                    if p.get("penalty") and kind == "Foul Committed"
                    else (
                        "Удаление"
                        if is_red
                        else "Жёлтая карточка"
                        if card
                        else "Автогол"
                        if kind == "Own Goal Against"
                        else "Нереализованный пенальти"
                        if e in penalty_shots
                        else "Фол у штрафной"
                        if e in dangerous_fouls
                        else "Фол"
                        if kind == "Foul Committed"
                        else "Неудачное действие на выходе"
                    )
                )
                incident_rows.append(
                    {
                        "match_id": match["match_id"],
                        "event_id": e["id"],
                        "player_id": pid,
                        "minute": e["minute"],
                        "second": e["second"],
                        "kind": label,
                        "shot_id": shot["id"] if shot else None,
                        "shot_xg": payload(shot)["statsbomb_xg"] if shot else 0.0,
                        "goal": int(payload(shot)["outcome"]["name"] == "Goal")
                        if shot
                        else int(kind == "Own Goal Against"),
                        "seconds_after": shot["minute"] * 60 + shot["second"] - e["minute"] * 60 - e["second"]
                        if shot
                        else None,
                        "provider_error": 0,
                        "primary_link": int(shot is not None),
                        "note": ("Удар после стандарта или продолжения по преимуществу" if shot else "")
                        + (f" · {minutes_after_red:.1f} мин после удаления" if is_red else ""),
                    }
                )
        active_opponent_shots = (
            [
                e
                for e in raw
                if e["type"]["name"] == "Shot"
                and e["team"]["id"] != record["team_id"]
                and pd.notna(record["start_index"])
                and record["start_index"] <= e["index"] < record["end_index"]
            ]
            if record["role"] == ROLES[0]
            else []
        )
        faced_penalties = [
            e for e in active_opponent_shots if payload(e).get("type", {}).get("name") == "Penalty"
        ]

        def saved(e):
            return payload(e)["outcome"]["name"] in {"Saved", "Saved to Post"}

        sot_faced = [e for e in active_opponent_shots if saved(e) or payload(e)["outcome"]["name"] == "Goal"]
        goalie_saves = sum(saved(e) for e in active_opponent_shots)
        low_xg_goals = [
            e
            for e in active_opponent_shots
            if payload(e)["outcome"]["name"] == "Goal" and payload(e)["statsbomb_xg"] <= 0.10
        ]
        for e in low_xg_goals if record["team_id"] == TEAM_ID else []:
            incident_rows.append(
                {
                    "match_id": match["match_id"],
                    "event_id": e["id"],
                    "player_id": pid,
                    "minute": e["minute"],
                    "second": e["second"],
                    "kind": "Пропущен удар с xG ≤ 0,10",
                    "shot_id": e["id"],
                    "shot_xg": payload(e)["statsbomb_xg"],
                    "goal": 1,
                    "seconds_after": 0.0,
                    "provider_error": 0,
                    "primary_link": 0,
                    "note": "Флаг для просмотра, не подтверждённая ошибка: PSxG отсутствует. Штраф не начисляется.",
                }
            )
        record.update(
            match_id=match["match_id"],
            date=match["date"],
            is_home=int(record["team_id"] == match["home_id"]),
            opponent=match["away"] if record["team_id"] == match["home_id"] else match["home"],
            score=f"{match['home_score']}:{match['away_score']}",
            start=int(record["start_index"] == 0),
            recoveries=won_recoveries,
            successful_interceptions=won_interceptions,
            nonpenalty_goals=nonpenalty_goals,
            npxg=npxg,
            finishing=nonpenalty_goals - npxg,
            shots_on_target=sum(
                payload(e)["outcome"]["name"] in {"Goal", "Saved", "Saved to Post"} for e in own_shots
            ),
            dribbles_won=sum(
                e["type"]["name"] == "Dribble" and payload(e).get("outcome", {}).get("name") == "Complete"
                for e in actions
            ),
            penalties_taken=len(penalty_shots),
            penalties_scored=sum(payload(e)["outcome"]["name"] == "Goal" for e in penalty_shots),
            penalties_missed=sum(payload(e)["outcome"]["name"] != "Goal" for e in penalty_shots),
            penalties_won=sum(
                e["type"]["name"] == "Foul Won" and payload(e).get("penalty", False) for e in actions
            ),
            offsides_received=sum(
                e["type"]["name"] == "Pass"
                and payload(e).get("outcome", {}).get("name") == "Pass Offside"
                and payload(e).get("recipient", {}).get("id") == pid
                for e in raw
            ),
            fouls=len(fouls_committed),
            dangerous_fouls=len(dangerous_fouls),
            penalties_conceded=p_conceded,
            fouls_to_shots=sum(fouls[e["id"]] is not None for e in fouls_committed),
            fouls_to_goals=sum(
                fouls[e["id"]] is not None and payload(fouls[e["id"]])["outcome"]["name"] == "Goal"
                for e in fouls_committed
            ),
            yellow_cards=yellow,
            red_cards=red,
            minutes_after_red=minutes_after_red,
            own_goals=sum(e["type"]["name"] == "Own Goal Against" for e in actions),
            losses_own_half=sum(
                e["type"]["name"] in {"Miscontrol", "Dispossessed"} and e.get("location", [120])[0] < 60
                for e in actions
            ),
            saves=goalie_saves,
            shots_faced_on_target=len(sot_faced),
            save_pct=100 * goalie_saves / len(sot_faced) if sot_faced else None,
            penalties_faced=len(faced_penalties),
            penalties_saved=sum(saved(e) for e in faced_penalties),
            penalties_allowed=sum(payload(e)["outcome"]["name"] == "Goal" for e in faced_penalties),
            low_xg_conceded=len(low_xg_goals),
            handling_failures=sum(
                e["type"]["name"] == "Goal Keeper"
                and payload(e).get("type", {}).get("name")
                in {"Collected", "Punch", "Smother", "Keeper Sweeper"}
                and payload(e).get("outcome", {}).get("name") == "Fail"
                for e in actions
            ),
            collections=sum(
                e["type"]["name"] == "Goal Keeper"
                and payload(e).get("type", {}).get("name") == "Collected"
                and payload(e).get("outcome", {}).get("name") in {"Success", "Collected Twice"}
                for e in actions
            ),
            sweeps=sum(
                e["type"]["name"] == "Goal Keeper"
                and payload(e).get("type", {}).get("name") == "Keeper Sweeper"
                and payload(e).get("outcome", {}).get("name") in {"Claim", "Clear", "Success"}
                for e in actions
            ),
            deduction=sum(value for value, _ in charges[pid].values()),
            deduction_detail=json.dumps(
                [
                    {"episode": key, "cost": round(value, 3), "reason": reason}
                    for key, (value, reason) in charges[pid].items()
                    if value > 0
                ],
                ensure_ascii=False,
            ),
        )
        output.append(record)
    return pd.DataFrame(output), pd.DataFrame(incident_rows)


def rate_matches(frame):
    frame = frame.copy()
    frame["rating_profile"] = np.where(
        frame.role.eq(ROLES[2])
        & (
            frame.position.str.contains("Attacking", na=False)
            | frame.position.isin(["Left Midfield", "Right Midfield"])
        ),
        "Атакующая полузащита",
        frame.role,
    )
    totals = frame.groupby("player_id").minutes.sum()
    frame["rating"] = np.nan
    frame["metric_score"] = np.nan
    references = {}
    for name, weights in WEIGHTS.items():
        reference = frame[
            frame.rating_profile.eq(name) & frame.minutes.ge(30) & frame.player_id.map(totals).ge(450)
        ]
        candidates = frame[frame.rating_profile.eq(name) & frame.minutes.ge(30)]
        if reference.empty:
            continue
        weighted = pd.Series(0.0, index=candidates.index)
        references[name] = {}
        for feature, weight in weights.items():
            baseline = (
                reference[feature] if feature in RATIOS else reference[feature] * 90 / reference.minutes
            )
            values = (
                candidates[feature] if feature in RATIOS else candidates[feature] * 90 / candidates.minutes
            )
            dist = np.sort(baseline.dropna().to_numpy())
            references[name][feature] = {
                "weight": weight,
                "rows": len(dist),
                "median": float(np.median(dist)),
            }
            percentiles = (
                np.searchsorted(dist, values, side="left") + np.searchsorted(dist, values, side="right")
            ) / (2 * len(dist))
            weighted += weight * np.where(values.isna(), 0.5, percentiles)
        base = 6 + 6 * (weighted - 0.5)
        bonus = (candidates.penalties_saved * 0.6 + candidates.penalties_won * 0.2).clip(upper=1)
        frame.loc[candidates.index, "metric_score"] = base
        frame.loc[candidates.index, "rating"] = (base + bonus - candidates.deduction).clip(0, 10)
    return frame, references


def aggregate_season(frame):
    """Sum counts, recompute ratios from denominators, minute-weight match ratings."""
    numeric = [
        c
        for c in frame.select_dtypes(include="number").columns
        if c
        not in {
            "match_id",
            "player_id",
            "team_id",
            "start_index",
            "end_index",
            "end",
            "rating",
            "metric_score",
            "completion_pct",
            "save_pct",
            "is_home",
        }
    ]
    output = []
    for pid, games in frame.groupby("player_id"):
        row = {key: float(games[key].sum()) for key in numeric}
        role_minutes = games.groupby("role").minutes.sum()
        eligible = games[games.rating.notna()]
        row.update(
            player_id=int(pid),
            player=games.player.iloc[0],
            name=NAMES.get(pid, games.player.iloc[0].replace("''", "'")),
            role=role_minutes.idxmax(),
            rating_profile=games.groupby("rating_profile").minutes.sum().idxmax(),
            matches=len(games),
            starts=int(games.start.sum()),
            rated_matches=len(eligible),
            rated_minutes=float(eligible.minutes.sum()),
            average_rating=float(np.average(eligible.rating, weights=eligible.minutes))
            if len(eligible)
            else None,
            completion_pct=100 * games.completed_passes.sum() / games.passes.sum()
            if games.passes.sum()
            else None,
            save_pct=100 * games.saves.sum() / games.shots_faced_on_target.sum()
            if games.shots_faced_on_target.sum()
            else None,
            reliable=int(games.minutes.sum() >= 450),
        )
        output.append(row)
    return pd.DataFrame(output)


def write_season_report(summary, catalog):
    lc = catalog[(catalog.home_id == TEAM_ID) | (catalog.away_id == TEAM_ID)].copy()
    lc["gf"] = np.where(lc.home_id.eq(TEAM_ID), lc.home_score, lc.away_score)
    lc["ga"] = np.where(lc.home_id.eq(TEAM_ID), lc.away_score, lc.home_score)
    wins, draws = int(lc.gf.gt(lc.ga).sum()), int(lc.gf.eq(lc.ga).sum())
    rows = [
        "# Leicester City 2015/16: вклад игроков за сезон",
        "",
        f"Матчи: {len(lc)} · очки: {wins * 3 + draws} · победы: {wins} · ничьи: {draws} · поражения: {len(lc) - wins - draws} · мячи: {lc.gf.sum()}:{lc.ga.sum()}.",
        "",
        "## Состав",
        "",
        "TL — собственный описательный индекс, не официальный рейтинг. Средняя взвешена по минутам матчей с 30+ минутами. Звёздочка — менее 450 минут за сезон.",
        "",
        "| Игрок | Роль | Матчи | Минуты | Средняя TL | Голы | Ассисты | xA |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in summary.sort_values("minutes", ascending=False).itertuples():
        rating = f"{p.average_rating:.2f}" if pd.notna(p.average_rating) else "—"
        rows.append(
            f"| {p.name}{'*' if not p.reliable else ''} | {p.role} | {p.matches} | {p.minutes:.1f} | {rating} | {p.goals:.0f} | {p.assists:.0f} | {p.xa:.2f} |"
        )
    rows.extend(
        [
            "",
            "## Лидеры по вкладу",
            "",
            "Сравнение сезонных сумм среди игроков с 450+ минутами; категории не объединяются в причинную оценку важности.",
            "",
            "| Категория | Игрок | Всего | На 90 минут |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    eligible = summary[summary.reliable.eq(1)]
    for feature, title in [
        ("goals", "Голы"),
        ("assists", "Ассисты"),
        ("xa", "xA"),
        ("recoveries", "Возвраты мяча"),
        ("successful_interceptions", "Успешные перехваты"),
        ("tackles_won", "Успешные отборы"),
    ]:
        top = eligible.loc[eligible[feature].idxmax()]
        rows.append(
            f"| {title} | {top['name']} | {top[feature]:.2f} | {top[feature] * 90 / top.minutes:.2f} |"
        )
    keeper = summary[summary.player_id.eq(3815)].iloc[0]
    rows.extend(
        [
            "",
            "## Вратарь",
            "",
            f"Каспер Шмейхель: {keeper.matches} матчей, {keeper.minutes:.0f} минут, {keeper.saves:.0f} сейва, {keeper.goals_conceded:.0f} пропущенных голов; доля отражённых ударов в створ {keeper.save_pct:.1f}%.",
            "",
            f"Голов после ударов с обычным xG≤0,10: {keeper.low_xg_conceded:.0f}. Это флаги для просмотра, а не число ошибок на лёгких ударах: PSxG отсутствует, автоматический дополнительный вычет за низкий xG не применяется.",
            "",
            "## Дисциплина и пенальти",
            "",
            "| Игрок | Фолы | У штрафной | Пенальти в свои | ЖК | КК | Незабитые пенальти | Фолы перед голом |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for p in summary[summary.fouls.gt(0)].sort_values("fouls", ascending=False).itertuples():
        rows.append(
            f"| {p.name} | {p.fouls:.0f} | {p.dangerous_fouls:.0f} | {p.penalties_conceded:.0f} | {p.yellow_cards:.0f} | {p.red_cards:.0f} | {p.penalties_missed:.0f} | {p.fouls_to_goals:.0f} |"
        )
    rows.extend(
        [
            "",
            "## Метод и ограничения",
            "",
            "Данные: StatsBomb Open Data, сезон АПЛ 2015/16; ассисты и события считаются по разметке этого провайдера и могут отличаться от иных источников. Голы игроков не включают автоголы соперников.",
            "",
            "Счётчики рассчитаны на 90 минут, проценты пересчитаны по общим знаменателям. Минуты ограничены основным временем. Контрольные совпадения: 38 матчей, 3420 минут и 36 пропущенных у Шмейхеля; 24 гола Варди.",
            "",
            "Оценка использует процентиль показателей среди сопоставимых матчей игроков АПЛ с 450+ минутами за сезон. Это описательная оценка с выбранными нами весами, а не предсказание и не экспертная разметка. Атакующая полузащита имеет отдельный профиль.",
            "",
            "Потери связываются с первым ударом в пределах 20 секунд; фолы — с первым ударом в том же владении со стандарта/преимущества до 180 секунд. Связи не доказывают единоличную вину. При пенальти, карточке и голе одного связанного эпизода применяется максимальный вычет.",
            "",
            "В проекте также есть отдельный ML-эксперимент собственной xG-модели (scikit-learn), временное разделение и сравнение с базой: см. research.md. LLM получает рассчитанные факты и проверяемые ссылки; живой вызов требует API-ключа.",
            "",
            "Воспроизведение: python -m touchline.download → python -m touchline.ingest → python -m touchline.season. Подробная формула: ../docs/leicester-rating.md.",
        ]
    )
    (REPORTS / "leicester-season.md").write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratings-only", action="store_true", help="Reuse existing player facts and recompute ratings"
    )
    args = parser.parse_args()
    directories()
    catalog = matches()
    frames, incident_frames = [], []
    for number, match in enumerate([] if args.ratings_only else catalog.to_dict("records"), 1):
        mid = match["match_id"]
        players, incidents = build_match(
            read_raw(f"events/{mid}.json"), match_events(mid), match_shots(mid), match
        )
        frames.append(players)
        if len(incidents):
            incident_frames.append(incidents)
        if number % 40 == 0 or number == len(catalog):
            print(f"Player matches: {number}/{len(catalog)}", flush=True)
    all_matches, references = rate_matches(
        query("SELECT * FROM player_matches") if args.ratings_only else pd.concat(frames, ignore_index=True)
    )
    scores = catalog.set_index("match_id")[["home_score", "away_score"]].to_dict("index")
    all_matches["score"] = all_matches.apply(
        lambda r: (
            f"{scores[r.match_id]['home_score']}:{scores[r.match_id]['away_score']}"
            if r.is_home
            else f"{scores[r.match_id]['away_score']}:{scores[r.match_id]['home_score']}"
        ),
        axis=1,
    )
    leicester = all_matches[all_matches.team_id.eq(TEAM_ID)].copy()
    summary = aggregate_season(leicester)
    episodes = (
        query("SELECT * FROM leicester_incidents")
        if args.ratings_only
        else pd.concat(incident_frames, ignore_index=True)
    )
    episodes.loc[episodes.kind.eq("Автогол"), "goal"] = 1
    with closing(sqlite3.connect(DB)) as conn:
        all_matches.to_sql("player_matches", conn, if_exists="replace", index=False)
        summary.to_sql("leicester_players", conn, if_exists="replace", index=False)
        episodes.to_sql("leicester_incidents", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_player_matches_player ON player_matches(player_id, date)"
        )
        conn.commit()
    summary.to_csv(ARTIFACTS / "leicester_players.csv", index=False)
    write_season_report(summary, catalog)
    leicester.to_csv(ARTIFACTS / "leicester_player_matches.csv", index=False)
    episodes.to_csv(ARTIFACTS / "leicester_incidents.csv", index=False)
    (ARTIFACTS / "season_meta.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "team": TEAM,
                "matches": int(leicester.match_id.nunique()),
                "players": len(summary),
                "reference_matches": len(catalog),
                "rating_weights": WEIGHTS,
                "reference": references,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        summary[["name", "role", "matches", "minutes", "average_rating"]]
        .sort_values("average_rating", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
