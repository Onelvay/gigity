"""Role-specific player facts and inspectable sequences after unsuccessful actions.

An unsuccessful action is not automatically a mistake or a cause of defeat.
Only Error is presented as the provider's explicit error annotation.
"""

from collections import Counter, defaultdict

import pandas as pd

ROLES = ["Вратари", "Защитники", "Полузащитники", "Нападающие"]
CONTROL = {"Pass", "Carry", "Dribble", "Shot"}
RESTARTS = {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in", "Throw In"}
INCIDENT_COLUMNS = [
    "event_id",
    "event_index",
    "team_id",
    "team",
    "player_id",
    "player",
    "role",
    "minute",
    "second",
    "clock",
    "kind",
    "under_pressure",
    "recipient",
    "x",
    "y",
    "provider_error",
    "shot_id",
    "shot_player",
    "shot_xg",
    "shot_goal",
    "seconds_after",
    "end_index",
    "primary_link",
    "observation",
]


def role(position):
    if position == "Goalkeeper":
        return ROLES[0]
    if "Back" in position:
        return ROLES[1]
    if "Midfield" in position:
        return ROLES[2]
    if "Forward" in position or "Wing" in position or "Striker" in position:
        return ROLES[3]
    return "Позиция не определена"


def player_roster(raw, events):
    indexed = events.set_index("event_id")
    end_clock = float(events.clock.max())
    end_index = int(events.event_index.max()) + 1
    roster, positions = {}, defaultdict(Counter)

    def register(player, team, position, start=None, index=None):
        if not player or "id" not in player:
            return
        pid = player["id"]
        if pid not in roster:
            roster[pid] = {
                "player_id": pid,
                "player": player["name"],
                "team_id": team["id"],
                "team": team["name"],
                "position": position,
                "start": start,
                "start_index": index,
                "end": end_clock,
                "end_index": end_index,
            }
        elif start is not None:
            roster[pid]["start"], roster[pid]["start_index"] = start, index

    for event in sorted(raw, key=lambda e: e["index"]):
        kind, team = event["type"]["name"], event["team"]
        clock = float(indexed.loc[event["id"], "clock"])
        if kind == "Starting XI":
            for entry in event["tactics"]["lineup"]:
                register(entry["player"], team, entry["position"]["name"], 0.0, 0)
        player = event.get("player")
        position = event.get("position", {}).get("name", "Unknown")
        register(player, team, position)
        if player and position != "Unknown" and kind not in {"Starting XI", "Tactical Shift", "Substitution"}:
            positions[player["id"]][position] += 1
        if kind == "Substitution":
            outgoing = roster[player["id"]]
            outgoing["end"], outgoing["end_index"] = clock, event["index"]
            register(event["substitution"]["replacement"], team, position, clock, event["index"])
        payload = event.get(kind.lower().replace(" ", "_"), {})
        if player and payload.get("card", {}).get("name") in {"Red Card", "Second Yellow"}:
            roster[player["id"]]["end"] = clock
            roster[player["id"]]["end_index"] = event["index"]
    for pid, record in roster.items():
        if positions[pid]:
            record["position"] = positions[pid].most_common(1)[0][0]
        record["role"] = role(record["position"])
        record["minutes"] = (
            round(max(0, record["end"] - record["start"]), 1) if record["start"] is not None else None
        )
    return pd.DataFrame(roster.values())


def incident_kind(event):
    kind = event["type"]["name"]
    if kind == "Error":
        return "Ошибка по разметке StatsBomb"
    if kind == "Miscontrol":
        return "Неудачная обработка"
    if kind == "Dispossessed":
        return "Мяч отобрали"
    if kind == "Dribbled Past":
        return "Обыгран соперником"
    if kind == "Dribble" and event.get("dribble", {}).get("outcome", {}).get("name") == "Incomplete":
        return "Неудачная обводка"
    if kind == "Pass":
        return {
            "Incomplete": "Неточная передача",
            "Out": "Передача за пределы поля",
            "Pass Offside": "Передача в офсайд",
        }.get(event.get("pass", {}).get("outcome", {}).get("name"))
    return None


def find_incidents(raw, events, roster, window=20):
    if not 1 <= window <= 60:
        raise ValueError("Review window must be between 1 and 60 seconds")
    ordered = sorted(raw, key=lambda e: e["index"])
    lookup = events.set_index("event_id").to_dict("index")
    people = roster.set_index("player_id").to_dict("index")
    records = []
    for offset, event in enumerate(ordered):
        category = incident_kind(event)
        if not category or "player" not in event:
            continue
        row, person = lookup[event["id"]], people[event["player"]["id"]]
        record = {
            key: row[key]
            for key in [
                "event_index",
                "team_id",
                "team",
                "player_id",
                "player",
                "minute",
                "second",
                "clock",
                "under_pressure",
                "recipient",
                "x",
                "y",
            ]
        }
        record.update(
            event_id=event["id"],
            role=person["role"],
            kind=category,
            provider_error=event["type"]["name"] == "Error",
            shot_id=None,
            shot_player=None,
            shot_xg=0.0,
            shot_goal=False,
            seconds_after=None,
            end_index=event["index"],
            primary_link=False,
            observation="Удар в заданном окне не обнаружен",
        )
        stopped = event.get("out", False) or category in {"Передача за пределы поля", "Передача в офсайд"}
        possession = None
        if stopped:
            record["observation"] = "После действия мяч вышел из игры или зафиксирован офсайд"
        else:
            for following in ordered[offset + 1 :]:
                next_row = lookup[following["id"]]
                elapsed = (next_row["clock"] - row["clock"]) * 60
                if following["period"] != event["period"] or elapsed > window:
                    break
                kind = following["type"]["name"]
                record["end_index"] = following["index"]
                if kind in {"Half End", "Foul Committed", "Foul Won", "Offside"}:
                    break
                if kind == "Pass" and following.get("pass", {}).get("type", {}).get("name") in RESTARTS:
                    break
                owner = following.get("possession_team", {}).get("id")
                if kind in CONTROL and following["team"]["id"] == row["team_id"] and owner == row["team_id"]:
                    record["observation"] = "Команда снова совершила контролируемое действие до удара"
                    break
                if (
                    kind in CONTROL
                    and following["team"]["id"] != row["team_id"]
                    and owner == following["team"]["id"]
                ):
                    if possession is None:
                        possession = following.get("possession")
                    elif following.get("possession") != possession:
                        break
                    if kind == "Shot":
                        shot = following["shot"]
                        record.update(
                            shot_id=following["id"],
                            shot_player=following["player"]["name"],
                            shot_xg=float(shot["statsbomb_xg"]),
                            shot_goal=shot["outcome"]["name"] == "Goal",
                            seconds_after=round(elapsed, 1),
                            observation="В эпизоде соперника после действия зарегистрирован удар",
                        )
                        break
                if following.get("out", False):
                    break
        records.append(record)
    result = pd.DataFrame(records, columns=INCIDENT_COLUMNS)
    # Several actions can precede one shot. Count the shot only against the latest
    # preceding action for the losing team; keep other actions available for review.
    linked = result[result.shot_id.notna()].sort_values("event_index")
    unique = linked.drop_duplicates(["team_id", "shot_id"], keep="last").index
    result.loc[unique, "primary_link"] = True
    return result


def analyze_players(raw, events, shots, window=20):
    roster = player_roster(raw, events)
    incidents = find_incidents(raw, events, roster, window)
    shot_lookup = {e["id"]: e for e in raw if e["type"]["name"] == "Shot"}
    key_pass_shots = {e["shot"]["key_pass_id"]: e for e in shot_lookup.values() if "key_pass_id" in e["shot"]}
    output = []
    for record in roster.to_dict("records"):
        actions = [e for e in raw if e.get("player", {}).get("id") == record["player_id"]]
        passes = [e for e in actions if e["type"]["name"] == "Pass"]
        completed = [e for e in passes if "outcome" not in e["pass"]]
        attempts = [
            e
            for e in actions
            if e["type"]["name"] == "Duel" and e.get("duel", {}).get("type", {}).get("name") == "Tackle"
        ]
        key_passes, xa = 0, 0.0
        for passed in passes:
            shot = shot_lookup.get(passed["pass"].get("assisted_shot_id")) or key_pass_shots.get(passed["id"])
            if shot or passed["pass"].get("shot_assist"):
                key_passes += 1
            if shot:
                xa += shot["shot"]["statsbomb_xg"]
        pshots = shots[shots.event_id.isin([e["id"] for e in actions])]
        own_incidents = incidents[incidents.player_id.eq(record["player_id"])]
        primary = own_incidents[own_incidents.primary_link.eq(True)]
        counts = Counter(e["type"]["name"] for e in actions)
        conceded = 0
        if record["role"] == ROLES[0] and record["start_index"] is not None:
            for e in raw:
                if not record["start_index"] <= e["index"] < record["end_index"]:
                    continue
                conceded += int(
                    (
                        e["type"]["name"] == "Shot"
                        and e["team"]["id"] != record["team_id"]
                        and e["shot"]["outcome"]["name"] == "Goal"
                    )
                    or (e["type"]["name"] == "Own Goal Against" and e["team"]["id"] == record["team_id"])
                )
        record.update(
            passes=len(passes),
            completed_passes=len(completed),
            completion_pct=round(len(completed) / len(passes) * 100, 1) if passes else None,
            progressive_passes=sum(
                e["pass"].get("end_location", [0])[0] - e["location"][0] >= 10
                for e in completed
                if "location" in e
            ),
            key_passes=key_passes,
            xa=round(xa, 3),
            assists=sum(e["pass"].get("goal_assist", False) for e in passes),
            shots=len(pshots),
            goals=int(pshots.goal.sum()),
            xg=round(float(pshots.statsbomb_xg.sum()), 3),
            missed_shot_xg=round(float(pshots.loc[pshots.goal.eq(0), "statsbomb_xg"].sum()), 3),
            tackles=len(attempts),
            tackles_won=sum(
                e["duel"].get("outcome", {}).get("name") in {"Won", "Success In Play", "Success Out"}
                for e in attempts
            ),
            interceptions=counts["Interception"],
            blocks=counts["Block"],
            clearances=counts["Clearance"],
            pressures=counts["Pressure"],
            dribbled_past=counts["Dribbled Past"],
            miscontrols=counts["Miscontrol"],
            dispossessed=counts["Dispossessed"],
            unsuccessful_passes=sum(incident_kind(e) is not None for e in passes),
            provider_errors=counts["Error"],
            shots_after=len(primary),
            goals_after=int(primary.shot_goal.sum()),
            xg_after=round(float(primary.shot_xg.sum()), 3),
            saves=sum(
                e["type"]["name"] == "Goal Keeper"
                and e.get("goalkeeper", {}).get("type", {}).get("name")
                in {"Shot Saved", "Shot Saved to Post"}
                for e in actions
            ),
            goals_conceded=conceded,
        )
        output.append(record)
    return pd.DataFrame(output), incidents


def role_findings(players):
    """Role-specific observations, rather than an invented universal match rating."""
    if players.empty:
        return []
    name = players.role.iloc[0]
    notes = []
    if name == ROLES[0]:
        notes.append(
            f"События сейвов: {int(players.saves.sum())}; пропущенные при нахождении в воротах: {int(players.goals_conceded.sum())}."
        )
        notes.append(
            "По обычному xG нельзя заключить, какие удары вратарь обязан был отразить. PSxG в этом наборе не используется."
        )
    elif name == ROLES[1]:
        notes.append(
            f"Успешные отборы: {int(players.tackles_won.sum())} из {int(players.tackles.sum())}; блокировки: {int(players.blocks.sum())}; выносы: {int(players.clearances.sum())}."
        )
        notes.append(
            f"Обыграны соперником в {int(players.dribbled_past.sum())} зарегистрированных эпизодах. Сам факт обыгрыша не доказывает позиционную ошибку."
        )
    elif name == ROLES[2]:
        notes.append(
            f"Передачи под удар: {int(players.key_passes.sum())}; xA: {players.xa.sum():.2f}; точные передачи вперёд на ≥10 единиц: {int(players.progressive_passes.sum())}."
        )
        notes.append(
            f"Неудачные обработки: {int(players.miscontrols.sum())}; мяч отобран в {int(players.dispossessed.sum())} эпизодах."
        )
    else:
        notes.append(
            f"Голы: {int(players.goals.sum())}; удары: {int(players.shots.sum())}; суммарный xG: {players.xg.sum():.2f}."
        )
        notes.append(
            f"xG незабитых ударов: {players.missed_shot_xg.sum():.2f}. Даже большой xG одного момента не означает, что промах был грубой ошибкой."
        )
    notes.append(
        f"Явных отметок Error от провайдера: {int(players.provider_errors.sum())}. После выделенных действий: {int(players.shots_after.sum())} ударов соперника, из них {int(players.goals_after.sum())} голов (временная связь)."
    )
    return notes
