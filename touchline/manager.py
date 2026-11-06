"""Manager-facing factual signals: separate contribution from review priority.

Colours mark individual metrics/events, not a verdict that a player is good/bad.
"""

import json

import numpy as np
import pandas as pd

from touchline.players import ROLES
from touchline.season import aggregate_season

COLOURS = {
    "green": "background-color:#123c2c;color:#b1f2c5",
    "yellow": "background-color:#493817;color:#ffe19b",
    "red": "background-color:#4b222a;color:#ffb8bf",
    "grey": "color:#9caec3",
}
BADGES = {
    "green": "🟢 Сильная сторона",
    "yellow": "🟡 Проверить",
    "red": "🔴 Приоритет разбора",
    "grey": "⚪ Мало данных",
}
POSITIVE = {
    ROLES[0]: ["save_pct", "completion_pct", "sweeps"],
    ROLES[1]: ["tackles_won", "successful_interceptions", "blocks", "recoveries", "completion_pct"],
    ROLES[2]: [
        "recoveries",
        "tackles_won",
        "successful_interceptions",
        "progressive_passes",
        "key_passes",
        "xa",
    ],
    "Атакующая полузащита": [
        "nonpenalty_goals",
        "assists",
        "xa",
        "key_passes",
        "dribbles_won",
        "progressive_passes",
    ],
    ROLES[3]: ["nonpenalty_goals", "npxg", "xa", "key_passes", "shots_on_target"],
}
TITLES = {
    "save_pct": "Доля сейвов",
    "completion_pct": "Точность передач",
    "sweeps": "Успешные выходы за мячом",
    "tackles_won": "Успешные отборы",
    "successful_interceptions": "Перехваты",
    "blocks": "Блокировки",
    "recoveries": "Возвраты мяча",
    "progressive_passes": "Передачи вперёд",
    "key_passes": "Пасы под удар",
    "xa": "xA",
    "nonpenalty_goals": "Голы без пенальти",
    "assists": "Голевые передачи",
    "dribbles_won": "Успешные обводки",
    "npxg": "xG без пенальти",
    "shots_on_target": "Удары в створ",
    "losses_own_half": "Потери при приёме/отборе в своей половине",
}
ACTIONS = {
    "recoveries": "Использовать как опору при возврате мяча.",
    "tackles_won": "Сохранить акцент на единоборствах при отборе мяча.",
    "successful_interceptions": "Разобрать удачные перехваты при подготовке игры без мяча.",
    "progressive_passes": "Использовать для продвижения мяча передачами.",
    "key_passes": "Задействовать при создании моментов.",
    "xa": "Сохранить роль в подготовке ударов партнёров.",
    "nonpenalty_goals": "Использовать сильную сторону в завершении атак.",
    "assists": "Сохранить связи с завершающими атаки игроками.",
    "dribbles_won": "Использовать для обыгрыша один в один.",
    "save_pct": "Сравнить сейвы по видео: показатель без поправки на сложность ударов.",
    "completion_pct": "Сохранить успешные способы розыгрыша мяча.",
    "sweeps": "Проверить удачные выходы при разборе игры за пределами ворот.",
    "blocks": "Разобрать удачные блокировки; объём не равен качеству позиционной игры.",
    "npxg": "Сохранить участие в создании и завершении качественных моментов.",
    "shots_on_target": "Сохранить варианты выхода на удар в створ.",
}


def percentile(value, values):
    values = pd.Series(values).dropna().to_numpy()
    if not len(values) or pd.isna(value):
        return None
    return float((np.sum(values < value) + 0.5 * np.sum(values == value)) / len(values))


def build_manager(players, player_matches, episodes):
    peers = aggregate_season(player_matches)
    peers = peers[peers.reliable.eq(1)]
    signals, summaries = [], []
    for p in players.to_dict("records"):
        pid = p["player_id"]
        own = episodes[episodes.player_id.eq(pid)]
        reference = peers[peers.rating_profile.eq(p["rating_profile"])]
        current = []

        def add(
            colour, metric, fact, action, source=None, rank=None, own_events=own, target=current, person=p
        ):
            source = own_events.iloc[:0] if source is None else source
            target.append(
                {
                    "player_id": person["player_id"],
                    "name": person["name"],
                    "role": person["role"],
                    "colour": colour,
                    "badge": BADGES[colour],
                    "metric": metric,
                    "fact": fact,
                    "action": action,
                    "percentile": rank,
                    "event_ids": json.dumps(source.event_id.unique().tolist()),
                    "match_ids": json.dumps([int(v) for v in source.match_id.unique()]),
                }
            )

        if p["reliable"]:
            for feature in POSITIVE[p["rating_profile"]]:
                ratio = feature.endswith("_pct")
                value = p[feature] if ratio else p[feature] * 90 / p["minutes"]
                population = reference[feature] if ratio else reference[feature] * 90 / reference.minutes
                rank = percentile(value, population)
                if rank is None or rank < 0.75 or p[feature] <= 0:
                    continue
                unit = "%" if ratio else " / 90"
                total = f"{p[feature]:.2f}" if feature in {"xa", "npxg"} else f"{p[feature]:.0f}"
                fact = f"{TITLES[feature]}: {value:.2f}{unit} · медиана роли {population.median():.2f}{unit}"
                if not ratio:
                    fact += f" · за сезон {total}"
                if feature == "save_pct":
                    fact += " · без PSxG"
                add("green", feature, fact, ACTIONS[feature], rank=rank)
            if p["passes"] >= 100 and pd.notna(p["completion_pct"]):
                rank = percentile(p["completion_pct"], reference.completion_pct)
                if rank is not None and rank < 0.25:
                    add(
                        "yellow",
                        "completion_pct",
                        f"Точность передач: {p['completion_pct']:.1f}% · медиана роли {reference.completion_pct.median():.1f}%",
                        "Проверить направления и сложность передач; не считать каждый неточный пас ошибкой.",
                    )
            if p["losses_own_half"] >= 5:
                rate = p["losses_own_half"] * 90 / p["minutes"]
                population = reference.losses_own_half * 90 / reference.minutes
                rank = percentile(rate, population)
                if rank is not None and rank >= 0.75:
                    add(
                        "yellow",
                        "losses_own_half",
                        f"Приём/отбор в своей половине: {int(p['losses_own_half'])} потерь · {rate:.2f}/90 · медиана {population.median():.2f}/90",
                        "Разобрать приём мяча и потери под давлением на своей половине.",
                        own[own.kind.isin(["Неудачная обработка", "Мяч отобрали"]) & own.event_x.lt(60)]
                        if "event_x" in own
                        else own.iloc[:0],
                    )
            if p["shots"] >= 20 and p["finishing"] <= -2:
                add(
                    "yellow",
                    "finishing",
                    f"Реализация без пенальти: {int(p['nonpenalty_goals'])} гола при npxG {p['npxg']:.2f} · разница {p['finishing']:+.2f}",
                    "Разобрать незабитые удары и работу над завершением; это не доказательство слабого навыка.",
                )
            if p["dangerous_fouls"] >= 3:
                rate = p["dangerous_fouls"] * 90 / p["minutes"]
                population = reference.dangerous_fouls * 90 / reference.minutes
                rank = percentile(rate, population)
                if rank is not None and rank >= 0.75:
                    source = own[own.kind.eq("Фол у штрафной")]
                    if {"event_x", "event_y", "event_type"}.issubset(own.columns):
                        source = own[
                            own.event_type.eq("Foul Committed")
                            & own.event_x.le(30)
                            & own.event_y.between(15, 65)
                        ]
                    add(
                        "yellow",
                        "dangerous_fouls",
                        f"У своей штрафной: {int(p['dangerous_fouls'])} фолов · {rate:.2f}/90 · медиана {population.median():.2f}/90",
                        "Разобрать дистанцию и выбор момента при единоборствах около штрафной.",
                        source,
                    )
            if p["yellow_cards"] >= 5:
                source = (
                    own[own.event_card.isin(["Yellow Card", "Second Yellow"])]
                    if "event_card" in own
                    else own[own.kind.eq("Жёлтая карточка")]
                )
                add(
                    "yellow",
                    "yellow_cards",
                    f"Жёлтые карточки: {int(p['yellow_cards'])}",
                    "Разобрать повторяющиеся поводы для предупреждений и риск удаления.",
                    source,
                )
        else:
            add(
                "grey",
                "minutes",
                f"Всего {p['minutes']:.1f} минуты · порог сравнения 450",
                "Не делать вывод о стабильности игрока по этой выборке.",
            )

        rules = [
            (
                "penalties_conceded",
                "red",
                "Пенальти в свои ворота",
                "Разобрать фолы в штрафной и отработать такие единоборства.",
                ["Пенальти в свои ворота"],
            ),
            (
                "red_cards",
                "red",
                "Удаления",
                "Разобрать решения в эпизодах с карточками и действия после первого предупреждения.",
                ["Удаление"],
            ),
            (
                "own_goals",
                "red",
                "Автоголы",
                "Разобрать конкретные касания и взаимодействие с вратарём по видео.",
                ["Автогол"],
            ),
            (
                "penalties_missed",
                "red" if p["penalties_missed"] >= 2 else "yellow",
                "Незабитые пенальти",
                "Разобрать исполнения перед выбором пенальтиста; число попыток невелико.",
                ["Нереализованный пенальти"],
            ),
            (
                "handling_failures",
                "yellow",
                "Неудачные выходы",
                "Разобрать выбор момента и работу с мячом на выходе.",
                ["Неудачное действие на выходе"],
            ),
        ]
        for feature, colour, title, action, kinds in rules:
            if not p[feature]:
                continue
            source = own[own.kind.isin(kinds)]
            fact = f"{title}: {int(p[feature])}"
            if feature == "red_cards":
                fact += f" · {p['minutes_after_red']:.1f} мин после удалений"
            if feature == "penalties_missed":
                fact += f" · забито {int(p['penalties_scored'])}/{int(p['penalties_taken'])}"
            if len(source):
                first = source.sort_values("date").iloc[0]
                fact += f" · пример {first.date}, {int(first.minute)}:{int(first.second):02d}"
                if "opponent" in first and "score" in first:
                    fact += f" · {first.opponent} {first.score}"
            add(colour, feature, fact, action, source)
        marked = own[own.provider_error.eq(1)]
        if p["provider_errors"]:
            scoring = marked[marked.goal.eq(1)]
            add(
                "red" if len(scoring) else "yellow",
                "provider_errors",
                f"Явные Error провайдера: {int(p['provider_errors'])} · с последующим голевым ударом {scoring.shot_id.nunique()}",
                "Открыть отмеченные ошибки и разобрать действия до удара.",
                marked,
            )
        linked = own[own.primary_link.eq(1) & own.goal.eq(1) & own.shot_id.notna()].drop_duplicates("shot_id")
        if len(linked):
            add(
                "red" if len(linked) >= 2 else "yellow",
                "linked_goals",
                f"После действий: {len(linked)} голевых ударов соперника · временная связь",
                "Разобрать всю цепочку атаки и действия остальных игроков; не назначать единственного виновника.",
                linked,
            )
        if p["role"] == ROLES[0] and p["low_xg_conceded"]:
            add(
                "yellow",
                "low_xg_conceded",
                f"Пропущено после ударов с xG≤0,10: {int(p['low_xg_conceded'])} · сложность сейвов неизвестна",
                "Проверить исполнение ударов по видео/PSxG. Не считать эти голы автоматически ошибками.",
                own[own.kind.eq("Пропущен удар с xG ≤ 0,10")],
            )
        positive = sorted(
            [s for s in current if s["colour"] == "green"], key=lambda s: s["percentile"], reverse=True
        )
        risks = sorted(
            [s for s in current if s["colour"] in {"red", "yellow"}],
            key=lambda s: (s["colour"] != "red", s["metric"] == "linked_goals"),
        )
        severity = (
            "red"
            if any(s["colour"] == "red" for s in risks)
            else "yellow"
            if risks
            else "green"
            if p["reliable"]
            else "grey"
        )
        strength_brief = "Мало минут" if not p["reliable"] else "Нет выраженного сигнала"
        if positive:
            metric = positive[0]["metric"]
            digits = 2 if metric in {"xa", "npxg"} else 1 if metric.endswith("_pct") else 0
            strength_brief = f"{TITLES[metric]}: {p[metric]:.{digits}f}"
            if not metric.endswith("_pct"):
                strength_brief += f" · {p[metric] * 90 / p['minutes']:.2f}/90"
            else:
                strength_brief += "%"
        summaries.append(
            {
                "player_id": pid,
                "name": p["name"],
                "role": p["role"],
                "minutes": p["minutes"],
                "contribution_colour": "green" if positive else "grey",
                "contribution": "🟢 Сильный вклад"
                if len(positive) >= 2
                else "🟢 Сильная сторона"
                if positive
                else "⚪ Мало минут"
                if not p["reliable"]
                else "⚪ Нет сильного сигнала",
                "strength": positive[0]["fact"]
                if positive
                else "Нет верхнего квартиля по выбранным метрикам"
                if p["reliable"]
                else "Менее 450 минут",
                "risk_colour": severity,
                "risk": BADGES[severity] if risks or not p["reliable"] else "🟢 Нет отмеченных рисков",
                "reason": risks[0]["fact"] if risks else "По правилам проверки значимых флагов нет",
                "action": risks[0]["action"]
                if risks
                else positive[0]["action"]
                if positive
                else "Наблюдать; проверить роль и объём игровых минут.",
                "strength_count": len(positive),
                "strength_brief": strength_brief,
                "reason_brief": risks[0]["fact"].split(" · пример")[0].split(" · медиана")[0]
                if risks
                else "По правилам значимых флагов нет",
            }
        )
        signals.extend(current)
    return pd.DataFrame(summaries), pd.DataFrame(
        signals,
        columns=[
            "player_id",
            "name",
            "role",
            "colour",
            "badge",
            "metric",
            "fact",
            "action",
            "percentile",
            "event_ids",
            "match_ids",
        ],
    )


def load_manager():
    from touchline.data import query

    players = query("SELECT * FROM leicester_players")
    games = query("SELECT * FROM player_matches")
    episodes = query(
        "SELECT i.*, m.date, CASE WHEN m.home_id=22 THEN m.away ELSE m.home END AS opponent, "
        "printf('%d:%d',CASE WHEN m.home_id=22 THEN m.home_score ELSE m.away_score END, "
        "CASE WHEN m.home_id=22 THEN m.away_score ELSE m.home_score END) AS score, "
        "e.x AS event_x, e.y AS event_y, e.type AS event_type, e.card AS event_card FROM leicester_incidents i "
        "JOIN matches m USING(match_id) LEFT JOIN events e USING(event_id)"
    )
    return build_manager(players, games, episodes)


if __name__ == "__main__":
    from touchline.config import ARTIFACTS

    summary, signals = load_manager()
    summary.to_csv(ARTIFACTS / "leicester_manager_summary.csv", index=False)
    signals.to_csv(ARTIFACTS / "leicester_manager_signals.csv", index=False)
    print(summary[["name", "contribution", "strength", "risk", "reason"]].to_string(index=False))
