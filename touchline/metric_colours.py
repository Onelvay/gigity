"""A colour and explanation for every displayed metric; no forced good/bad verdict."""

import pandas as pd

from touchline.manager import percentile
from touchline.players import ROLES

STYLES = {
    "green": "background-color:#123c2c;color:#b1f2c5",
    "yellow": "background-color:#493817;color:#ffe19b",
    "red": "background-color:#4b222a;color:#ffb8bf",
    "blue": "background-color:#142e44;color:#b8d8f5",
    "grey": "background-color:#232a33;color:#abb6c4",
}
BADGES = {
    "green": "🟢 Сильный / нет события",
    "yellow": "🟡 Внимание",
    "red": "🔴 Приоритет",
    "blue": "🔵 Норма / контекст",
    "grey": "⚪ Недостаточно данных",
}
BENEFICIAL = {
    "average_rating",
    "goal_involvements",
    "recoveries",
    "tackles_won",
    "successful_interceptions",
    "completion_pct",
    "progressive_passes",
    "key_passes",
    "xa",
    "goals",
    "nonpenalty_goals",
    "npxg",
    "finishing",
    "shots_on_target",
    "assists",
    "dribbles_won",
    "xg",
}
ADVERSE = {
    "miscontrols",
    "dispossessed",
    "unsuccessful_passes",
    "losses_own_half",
    "dribbled_past",
    "fouls",
    "dangerous_fouls",
    "yellow_cards",
    "shots_after",
    "goals_after",
    "xg_after",
    "fouls_to_shots",
    "fouls_to_goals",
}
KEEPER = {
    "saves",
    "save_pct",
    "goals_conceded",
    "penalties_saved",
    "penalties_allowed",
    "handling_failures",
    "low_xg_conceded",
    "penalties_faced",
    "shots_faced_on_target",
    "collections",
    "sweeps",
}
CRITICAL = {"penalties_conceded", "red_cards", "own_goals", "minutes_after_red"}


def classify_metric(player, key, peers, signals=None):
    value = player[key]
    if key in KEEPER and player["role"] != ROLES[0]:
        return "grey", "Вратарский показатель не применяется к этой роли."
    if pd.isna(value):
        return "grey", "Нет значения или ненулевого знаменателя."
    if key in {"penalties_scored", "penalties_missed"} and player["penalties_taken"] == 0:
        return "blue", "Игрок не исполнял пенальти; оценивать реализацию нельзя."
    if key in {"penalties_saved", "penalties_allowed"} and player["penalties_faced"] == 0:
        return "blue", "Пенальти по воротам этого вратаря не было."
    if key == "low_xg_conceded":
        return (
            "yellow" if value else "blue"
        ), "Флаг для просмотра. Низкий обычный xG не доказывает лёгкий сейв или ошибку; PSxG отсутствует."
    if key in CRITICAL:
        return ("red" if value > 0 else "green"), (
            "Есть значимый негативный эпизод для разбора."
            if value
            else "Таких событий не зарегистрировано; это не общая оценка игрока."
        )
    if signals is not None and len(signals):
        marked = signals[signals.metric.eq(key)]
        if marked.colour.eq("red").any():
            return "red", marked.loc[marked.colour.eq("red"), "fact"].iloc[0]
    if key in {"penalties_missed", "provider_errors", "handling_failures"}:
        if not value:
            return "green", "Таких событий не зарегистрировано."
        return (
            "red" if key == "penalties_missed" and value >= 2 else "yellow"
        ), "Есть незавершённые исполнения или отмеченные неудачные действия; откройте эпизоды."
    if key in {"penalties_saved", "penalties_scored", "penalties_won"}:
        return (
            "green" if value > 0 else "blue"
        ), "Положительный зарегистрированный результат; число попыток и контекст учитывайте отдельно."
    if key not in BENEFICIAL | ADVERSE | {"save_pct"}:
        return "blue", "Объём или контекст: больше/меньше само по себе не означает лучше/хуже."
    if player["minutes"] < 450:
        return "grey", "Менее 450 минут: сравнительная оценка не выставляется."
    reference = peers[peers.rating_profile.eq(player["rating_profile"]) & peers.minutes.ge(450)]
    if reference.empty:
        return "grey", "Нет сопоставимых игроков этой роли."
    ratio = key.endswith("_pct") or key == "average_rating"
    actual = value if ratio else value * 90 / player["minutes"]
    population = reference[key] if ratio else reference[key] * 90 / reference.minutes
    rank = percentile(actual, population)
    if rank is None:
        return "grey", "У сопоставимых игроков недостаточно значений."
    suffix = "/10" if key == "average_rating" else "%" if ratio else "/90"
    reason = (
        f"{actual:.2f}{suffix}; медиана роли {population.median():.2f}{suffix}; процентиль {rank * 100:.0f}."
    )
    if key == "save_pct":
        reason += " Без поправки на сложность ударов; это не оценка ошибок вратаря."
    if key in ADVERSE:
        if value == 0:
            return "green", "Таких событий не зарегистрировано."
        return (
            "yellow" if rank >= 0.75 else "blue"
        ), reason + " Для этого счётчика высокая частота — повод проверить эпизоды."
    if value == 0 and population.fillna(0).eq(0).all():
        return "blue", "Ноль у всех сопоставимых игроков: оснований выделять показатель нет."
    return ("green" if rank >= 0.75 else "yellow" if rank < 0.25 else "blue"), reason


def colour_table(player, keys, peers, signals=None):
    return pd.DataFrame(
        [
            {
                "metric": key,
                "colour": colour,
                "status": (
                    "🟢 Сильный показатель"
                    if key in BENEFICIAL or key == "save_pct"
                    else "🟢 Положительный факт"
                    if key in {"penalties_saved", "penalties_scored", "penalties_won"}
                    else "🟢 Нет события"
                )
                if colour == "green"
                else BADGES[colour],
                "reason": reason,
            }
            for key in keys
            for colour, reason in [classify_metric(player, key, peers, signals)]
        ]
    )
