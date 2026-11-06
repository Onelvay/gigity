"""Actual starting lineups, schematic positions and role-specific comparisons."""

import base64
from collections import Counter, defaultdict
from html import escape

import pandas as pd
import plotly.graph_objects as go

from touchline.config import ROOT
from touchline.download import read_raw
from touchline.metric_colours import colour_table
from touchline.season import NAMES, TEAM_ID, aggregate_season

PITCH_SOURCE = "https://commons.wikimedia.org/wiki/File:Football_field.svg"
TACTICS_SOURCE = "https://www.premierleague.com/en/news/2748139"
SLOTS = {
    1: (50, 135, "Вратарь", "ВР"),
    2: (87, 109, "Правый защитник", "ПЗ"),
    3: (64, 109, "Правый центральный защитник", "ЦЗ"),
    4: (50, 109, "Центральный защитник", "ЦЗ"),
    5: (36, 109, "Левый центральный защитник", "ЦЗ"),
    6: (13, 109, "Левый защитник", "ЛЗ"),
    7: (91, 93, "Правый латераль", "ПЛ"),
    8: (9, 93, "Левый латераль", "ЛЛ"),
    9: (64, 84, "Правый опорный полузащитник", "ОП"),
    10: (50, 84, "Опорный полузащитник", "ОП"),
    11: (36, 84, "Левый опорный полузащитник", "ОП"),
    12: (86, 65, "Правый полузащитник", "ПП"),
    13: (64, 65, "Правый центральный полузащитник", "ЦП"),
    14: (50, 65, "Центральный полузащитник", "ЦП"),
    15: (36, 65, "Левый центральный полузащитник", "ЦП"),
    16: (14, 65, "Левый полузащитник", "ЛП"),
    17: (87, 39, "Правый вингер", "ПВ"),
    18: (66, 44, "Правый атакующий полузащитник", "АП"),
    19: (50, 44, "Центральный атакующий полузащитник", "АП"),
    20: (34, 44, "Левый атакующий полузащитник", "АП"),
    21: (13, 39, "Левый вингер", "ЛВ"),
    22: (66, 22, "Правый нападающий", "НП"),
    23: (50, 22, "Центральный нападающий", "НП"),
    24: (34, 22, "Левый нападающий", "НП"),
    25: (50, 40, "Второй нападающий", "ВН"),
}
ROLE_KEYS = {
    "Вратарь": ["save_pct", "saves", "completion_pct", "handling_failures"],
    "Центральная защита": [
        "tackles_won",
        "successful_interceptions",
        "blocks",
        "dribbled_past",
        "dangerous_fouls",
        "penalties_conceded",
        "red_cards",
    ],
    "Крайняя защита": [
        "tackles_won",
        "successful_interceptions",
        "progressive_passes",
        "xa",
        "dangerous_fouls",
        "penalties_conceded",
        "red_cards",
    ],
    "Опорная полузащита": [
        "recoveries",
        "tackles_won",
        "successful_interceptions",
        "losses_own_half",
        "progressive_passes",
    ],
    "Центральная полузащита": ["progressive_passes", "completion_pct", "key_passes", "xa", "recoveries"],
    "Фланговая полузащита": ["key_passes", "xa", "assists", "dribbles_won", "completion_pct"],
    "Атакующая полузащита": ["xa", "key_passes", "nonpenalty_goals", "progressive_passes"],
    "Вингер": ["nonpenalty_goals", "xa", "dribbles_won", "key_passes"],
    "Нападение": ["nonpenalty_goals", "npxg", "finishing", "shots_on_target", "xa"],
}
ROLE_NOTES = {
    "Вратарь": "Оцениваем отражённые удары, выходы и начало атаки. Число пропущенных голов без PSxG не определяет ошибки на линии.",
    "Центральная защита": "В этой позиции первыми проверяем отборы, перехваты, блокировки, обыгрыши и фолы перед своими воротами.",
    "Крайняя защита": "Сопоставляем защиту фланга с продвижением мяча. Объём атакующих передач зависит от задания на матч.",
    "Опорная полузащита": "Приоритет — возврат мяча, перехваты и потери перед своей защитой. Мало передач под удар само по себе не означает плохую игру опорника.",
    "Центральная полузащита": "Смотрим на связь обороны с атакой: продвижение передачами, сохранение мяча и подготовку ударов.",
    "Фланговая полузащита": "В 4‑4‑2 проверяем создание моментов и работу с фланга. Точность передач нужно читать с учётом навесов, длины и риска; низкий процент не доказывает техническую ошибку.",
    "Атакующая полузащита": "Для разбора важнее подготовка ударов и участие в завершении, чем объём отборов.",
    "Вингер": "Проверяем создание моментов, обыгрыши и участие в завершении; это отдельная группа сравнения.",
    "Нападение": "Сопоставляем голы с качеством моментов и ударами в створ. Функцию второго форварда без мяча нельзя полностью оценить только по голам.",
}


def role_family(position):
    if position == "Goalkeeper":
        return "Вратарь"
    if "Center Back" in position:
        return "Центральная защита"
    if "Back" in position:
        return "Крайняя защита"
    if "Defensive Midfield" in position:
        return "Опорная полузащита"
    if "Attacking Midfield" in position:
        return "Атакующая полузащита"
    if position in {"Left Midfield", "Right Midfield"}:
        return "Фланговая полузащита"
    if "Midfield" in position:
        return "Центральная полузащита"
    if "Wing" in position:
        return "Вингер"
    return "Нападение"


def build_tactics(catalog):
    catalog = catalog[(catalog.home_id == TEAM_ID) | (catalog.away_id == TEAM_ID)].sort_values(
        ["date", "match_id"]
    )
    lineups, formations, patterns = {}, Counter(), defaultdict(lambda: {"shots": 0, "goals": 0, "xg": 0.0})
    starts, positional_starts = Counter(), defaultdict(Counter)
    for m in catalog.itertuples():
        events = read_raw(f"events/{m.match_id}.json")
        xi = next(e for e in events if e["team"]["id"] == TEAM_ID and e["type"]["name"] == "Starting XI")
        lineup = []
        for item in xi["tactics"]["lineup"]:
            p, pos = item["player"], item["position"]
            x, y, title, short = SLOTS[pos["id"]]
            lineup.append(
                {
                    "player_id": p["id"],
                    "name": NAMES.get(p["id"], p["name"]),
                    "position_id": pos["id"],
                    "position": pos["name"],
                    "position_ru": title,
                    "short": short,
                    "family": role_family(pos["name"]),
                    "number": item.get("jersey_number"),
                    "x": x,
                    "y": y,
                }
            )
            starts[p["id"]] += 1
            positional_starts[p["id"]][pos["name"]] += 1
        formation = xi["tactics"]["formation"]
        formations[formation] += 1
        lineups[int(m.match_id)] = {
            "formation": str(formation),
            "lineup": lineup,
            "date": m.date,
            "opponent": m.away if m.home_id == TEAM_ID else m.home,
            "score": f"{m.home_score}:{m.away_score}"
            if m.home_id == TEAM_ID
            else f"{m.away_score}:{m.home_score}",
        }
        for e in events:
            if e["team"]["id"] == TEAM_ID and e["type"]["name"] == "Shot":
                bucket = patterns[e.get("play_pattern", {}).get("name", "Unknown")]
                bucket["shots"] += 1
                bucket["goals"] += int(e["shot"]["outcome"]["name"] == "Goal")
                bucket["xg"] += e["shot"]["statsbomb_xg"]
    groups = defaultdict(list)
    for mid, data in lineups.items():
        groups[tuple(sorted(p["player_id"] for p in data["lineup"]))].append(mid)
    common = max(groups.values(), key=len)
    exact = Counter(
        tuple(sorted((p["player_id"], p["position_id"]) for p in lineups[mid]["lineup"])) for mid in common
    )
    pattern = exact.most_common(1)[0][0]
    representative = next(
        mid
        for mid in common
        if tuple(sorted((p["player_id"], p["position_id"]) for p in lineups[mid]["lineup"])) == pattern
    )
    return {
        "lineups": lineups,
        "formations": dict(formations),
        "patterns": dict(patterns),
        "starts": dict(starts),
        "positional_starts": dict(positional_starts),
        "representative": representative,
        "common_xi_matches": len(common),
    }


def role_comparison(player_id, position, player_matches):
    family = role_family(position)
    subset = player_matches[player_matches.position.map(role_family).eq(family)]
    peers = aggregate_season(subset)
    peers["rating_profile"] = family
    selected = peers[peers.player_id.eq(player_id)]
    if selected.empty:
        return None, pd.DataFrame(), family
    player = selected.iloc[0]
    keys = ROLE_KEYS[family]
    coloured = colour_table(player, keys, peers)
    return player, coloured, family


def shirt_svg(number, selected=False, keeper=False):
    colour = "#d97706" if keeper else "#124bb1"
    stroke = "#fde047" if selected else "#c3d8ff"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="110" viewBox="0 0 100 110">'
        f'<path d="M28 10 L40 3 Q50 18 60 3 L72 10 L94 30 L80 49 L70 39 L70 104 L30 104 L30 39 L20 49 L6 30 Z" '
        f'fill="{colour}" stroke="{stroke}" stroke-width="{5 if selected else 1.5}"/>'
        f'<text x="50" y="72" text-anchor="middle" font-family="Arial" font-size="40" font-weight="bold" fill="white">{escape(str(number))}</text></svg>'
    )


def formation_figure(lineup, selected_id=None):
    field = (ROOT / "assets/football-field-public-domain.svg").read_bytes()
    fig = go.Figure()
    fig.add_layout_image(
        source="data:image/svg+xml;base64," + base64.b64encode(field).decode(),
        xref="x",
        yref="y",
        x=0,
        y=0,
        sizex=100,
        sizey=150,
        xanchor="left",
        yanchor="top",
        sizing="stretch",
        layer="below",
    )
    for p in lineup:
        icon = shirt_svg(p["number"], p["player_id"] == selected_id, p["position_id"] == 1)
        fig.add_layout_image(
            source="data:image/svg+xml;base64," + base64.b64encode(icon.encode()).decode(),
            xref="x",
            yref="y",
            x=p["x"],
            y=p["y"],
            sizex=12,
            sizey=13,
            xanchor="center",
            yanchor="middle",
            layer="below",
        )
        fig.add_annotation(
            x=p["x"],
            y=p["y"] + 9,
            text=f"<b>{escape(p['name'].split()[-1])}</b> · {p['short']}",
            showarrow=False,
            font=dict(size=11, color="white"),
            bgcolor="rgba(0,30,0,.75)",
            borderpad=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[p["x"] for p in lineup],
            y=[p["y"] for p in lineup],
            mode="markers",
            marker=dict(size=60, color="rgba(0,0,0,0)", line=dict(width=0)),
            customdata=[[p["player_id"], p["name"], p["position_ru"], p["number"]] for p in lineup],
            hovertemplate="%{customdata[1]}<br>%{customdata[2]} · №%{customdata[3]}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 100], visible=False, fixedrange=True)
    fig.update_yaxes(
        range=[150, 0],
        visible=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        constraintoward="top",
    )
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        clickmode="event+select",
        dragmode=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
