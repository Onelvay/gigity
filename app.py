"""Leicester City 2015/16: player cards and season tables, without charts."""

import json
import os
from html import escape

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import APIError

from touchline.config import ARTIFACTS, DB, ROOT, SOURCE_COMMIT
from touchline.data import query
from touchline.evidence import llm_report
from touchline.manager import COLOURS, load_manager
from touchline.metric_colours import STYLES, classify_metric, colour_table
from touchline.players import ROLES
from touchline.season import NAMES, TEAM_ID, WEIGHTS, aggregate_season
from touchline.tactics import (
    PITCH_SOURCE,
    ROLE_KEYS,
    ROLE_NOTES,
    TACTICS_SOURCE,
    build_tactics,
    formation_figure,
    role_comparison,
)

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Leicester · Сезон игроков", page_icon="🦊", layout="wide")
st.markdown(
    """<style>
.block-container{max-width:1180px;padding-top:3.8rem} h1{font-size:2rem!important;letter-spacing:-.04em}
h2{font-size:1.4rem!important} h3{font-size:1.1rem!important}
.season-label{color:#95a7bc;font-size:13px;letter-spacing:.12em}
[data-testid="stMetric"]{padding:12px 16px;border:1px solid #233043;border-radius:10px;background:#101b29}
[data-testid="stMetricValue"]{font-size:1.7rem} [data-testid="stMetricLabel"]{color:#9caec3}
.player-title{margin-top:16px;margin-bottom:3px;font-size:28px;font-weight:700}
.muted{color:#9caec3;font-size:13px}
@media(max-width:600px){
[data-testid="stPlotlyChart"]{max-height:calc((100vw - 32px)*1.5);overflow:hidden}
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]){max-height:calc((100vw - 32px)*1.5);overflow:hidden}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]){display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"])>[data-testid="stColumn"]{width:100%!important;min-width:0!important}
[data-testid="stMetric"]{padding:10px}[data-testid="stMetricValue"]{font-size:1.4rem}}
</style>""",
    unsafe_allow_html=True,
)
META = ARTIFACTS / "season_meta.json"
if not DB.exists() or not META.exists():
    st.title("Leicester City · 2015/16")
    st.info("Подготовьте данные сезона один раз.")
    st.code("python -m touchline.download\npython -m touchline.ingest\npython -m touchline.season")
    st.stop()


@st.cache_data
def load_season(version):
    return (
        query("SELECT * FROM leicester_players"),
        query("SELECT * FROM player_matches WHERE team_id=? ORDER BY date", (TEAM_ID,)),
        query("SELECT * FROM matches WHERE home_id=? OR away_id=? ORDER BY date", (TEAM_ID, TEAM_ID)),
        query("SELECT * FROM leicester_incidents"),
    )


@st.cache_data
def manager_data(version):
    return load_manager()


@st.cache_data
def metric_reference(version):
    return aggregate_season(query("SELECT * FROM player_matches"))


@st.cache_data
def tactical_data(version):
    return build_tactics(query("SELECT * FROM matches"))


@st.cache_data
def tactical_comparison(player_id, position, version):
    return role_comparison(player_id, position, query("SELECT * FROM player_matches"))


players, games, matches, episodes = load_season((DB.stat().st_mtime_ns, META.stat().st_mtime_ns))
manager_summary, manager_signals = manager_data((DB.stat().st_mtime_ns, META.stat().st_mtime_ns))
metric_peers = metric_reference((DB.stat().st_mtime_ns, META.stat().st_mtime_ns))
matches["opponent"] = matches.apply(lambda r: r.away if r.home_id == TEAM_ID else r.home, axis=1)
matches["venue"] = matches.home_id.map(lambda value: "Дома" if value == TEAM_ID else "В гостях")
matches["gf"] = matches.apply(lambda r: r.home_score if r.home_id == TEAM_ID else r.away_score, axis=1)
matches["ga"] = matches.apply(lambda r: r.away_score if r.home_id == TEAM_ID else r.home_score, axis=1)
matches["score"] = matches.gf.astype(str) + ":" + matches.ga.astype(str)
matches["result"] = matches.apply(lambda r: "В" if r.gf > r.ga else "Н" if r.gf == r.ga else "П", axis=1)
episodes = episodes.merge(
    matches[["match_id", "date", "opponent", "score"]], on="match_id", validate="many_to_one"
)
episodes["time"] = (
    episodes.minute.astype(int).astype(str) + ":" + episodes.second.map(lambda x: f"{int(x):02d}")
)
episodes["danger"] = (
    episodes.goal.eq(1)
    | episodes.provider_error.eq(1)
    | episodes.primary_link.eq(1)
    | episodes.kind.isin(
        [
            "Пенальти в свои ворота",
            "Удаление",
            "Автогол",
            "Нереализованный пенальти",
            "Неудачное действие на выходе",
        ]
    )
)
wins, draws, losses = (int(matches.result.eq(value).sum()) for value in ["В", "Н", "П"])
st.markdown('<div class="season-label">LEICESTER CITY · PREMIER LEAGUE 2015/16</div>', unsafe_allow_html=True)
st.title("Сезон в цифрах")
st.caption(
    f"{len(matches)} матчей · {wins * 3 + draws} очко · {wins} побед / {draws} ничьих / {losses} поражения · мячи {matches.gf.sum()}:{matches.ga.sum()}"
)
page = st.radio(
    "Раздел",
    ["Менеджеру", "Игрок", "Схема и роли", "Состав", "Матчи", "Оценки и данные"],
    horizontal=True,
    label_visibility="collapsed",
)

LABELS = {
    "recoveries": "Возвраты мяча",
    "tackles_won": "Успешные отборы",
    "tackles": "Попытки отбора",
    "successful_interceptions": "Успешные перехваты",
    "completion_pct": "Точность передач",
    "progressive_passes": "Точные передачи вперёд ≥10 ед.",
    "key_passes": "Пасы под удар",
    "xa": "xA",
    "losses_own_half": "Плохой приём / мяч отобрали в своей половине",
    "goals": "Голы",
    "nonpenalty_goals": "Голы без пенальти",
    "npxg": "xG без пенальти",
    "finishing": "Голы без пенальти − npxG",
    "shots": "Удары",
    "shots_on_target": "Удары в створ",
    "assists": "Голевые передачи",
    "saves": "Сейвы",
    "save_pct": "Отражено ударов в створ",
    "goals_conceded": "Пропущенные голы",
    "penalties_saved": "Отражённые пенальти",
    "penalties_allowed": "Пропущено с пенальти",
    "handling_failures": "Неудачные выходы по разметке",
    "unsuccessful_passes": "Незавершённые передачи",
    "low_xg_conceded": "Пропущено после ударов с xG ≤0,10",
    "blocks": "Блокировки",
    "clearances": "Выносы",
    "dribbled_past": "Соперник обыграл",
    "fouls": "Фолы",
    "dangerous_fouls": "Фолы у своей штрафной",
    "penalties_conceded": "Пенальти в свои ворота",
    "red_cards": "Удаления",
    "yellow_cards": "Жёлтые карточки",
    "miscontrols": "Неудачная обработка",
    "dispossessed": "Мяч отобрали",
    "provider_errors": "Явные ошибки StatsBomb",
    "shots_after": "Удары соперника после действий",
    "goals_after": "Голевые удары после действий",
    "xg_after": "xG ударов после действий",
    "penalties_missed": "Нереализованные пенальти",
    "penalties_won": "Заработанные пенальти",
    "own_goals": "Автоголы",
    "offsides_received": "Адресат передачи в офсайд",
    "fouls_to_goals": "Фолы перед голом со стандарта",
    "fouls_to_shots": "Фолы перед ударом со стандарта",
    "minutes_after_red": "Минут после удаления",
    "passes": "Передачи",
    "completed_passes": "Точные передачи",
    "dribbles_won": "Успешные обводки",
    "xg": "xG",
    "penalties_taken": "Пенальти исполнено",
    "penalties_scored": "Пенальти забито",
    "penalties_faced": "Пенальти против вратаря",
    "shots_faced_on_target": "Удары в створ по воротам",
    "collections": "Успешные ловли мяча",
    "sweeps": "Успешные выходы за мячом",
}
CORE = {
    ROLES[0]: [
        "saves",
        "save_pct",
        "goals_conceded",
        "penalties_saved",
        "penalties_allowed",
        "handling_failures",
        "unsuccessful_passes",
        "low_xg_conceded",
    ],
    ROLES[1]: [
        "tackles_won",
        "successful_interceptions",
        "blocks",
        "dribbled_past",
        "fouls",
        "dangerous_fouls",
        "penalties_conceded",
        "red_cards",
    ],
    ROLES[2]: [
        "recoveries",
        "tackles_won",
        "successful_interceptions",
        "completion_pct",
        "progressive_passes",
        "key_passes",
        "xa",
        "losses_own_half",
    ],
    ROLES[3]: ["goals", "nonpenalty_goals", "npxg", "finishing", "shots", "shots_on_target", "assists", "xa"],
}


def number(value, digits=0):
    return "—" if pd.isna(value) else f"{value:,.{digits}f}".replace(",", " ")


def metric_table(player, keys):
    records = []
    for key in keys:
        ratio = key.endswith("_pct")
        fractional = key in {"xa", "xg", "npxg", "finishing", "xg_after", "minutes_after_red"}
        total = number(player[key], 2 if fractional else 1 if ratio else 0) + (
            "%" if ratio and pd.notna(player[key]) else ""
        )
        records.append(
            {
                "Показатель": LABELS[key],
                "За сезон": total,
                "На 90 мин": "—"
                if ratio
                else number(player[key] * 90 / player.minutes, 2)
                if player.minutes
                else "—",
            }
        )
    return pd.DataFrame(records)


def coloured_signals(pid, key, compact=False):
    signals = manager_signals[manager_signals.player_id.eq(pid)]
    good = signals[signals.colour.eq("green")].sort_values("percentile", ascending=False)
    risks = signals[signals.colour.isin(["red", "yellow"])].copy()
    risks["priority"] = risks.colour.map({"red": 0, "yellow": 1})
    risks = risks.sort_values("priority", kind="stable")
    choices = pd.concat([good.head(1), risks.head(2)])
    if choices.empty:
        choices = signals.head(1)
    for signal in choices.itertuples():
        style = COLOURS[signal.colour]
        st.markdown(
            f'<div style="{style};border-radius:8px;padding:10px 14px;margin:6px 0">'
            f'<div style="font-size:12px">{escape(signal.badge)}</div>'
            f'<div style="font-weight:600;font-size:14px">{escape(signal.fact)}</div>'
            f'<div style="font-size:13px;margin-top:3px">{escape(signal.action)}</div></div>',
            unsafe_allow_html=True,
        )
    with st.expander("Все причины и подтверждающие эпизоды"):
        full = signals[["badge", "fact", "action"]].rename(
            columns={"badge": "Сигнал", "fact": "Цифры и причина", "action": "Что сделать"}
        )
        st.dataframe(full, hide_index=True, width="stretch")
        source_signals = signals[signals.event_ids.ne("[]")]
        if len(source_signals):
            options = dict(zip(source_signals.index.astype(int), source_signals.fact, strict=True))
            selected = st.selectbox(
                "Причина для проверки", options, format_func=options.get, key=f"signal_{key}"
            )
            signal = source_signals.loc[selected]
            selected_episodes = episodes[
                episodes.player_id.eq(pid) & episodes.event_id.isin(json.loads(signal.event_ids))
            ]
            selected_episodes = selected_episodes.sort_values(["date", "minute", "second"])
            st.dataframe(
                selected_episodes[["date", "opponent", "time", "kind", "shot_xg", "goal", "event_id"]].rename(
                    columns={
                        "date": "Дата",
                        "opponent": "Соперник",
                        "time": "Время",
                        "kind": "Действие",
                        "shot_xg": "xG удара",
                        "goal": "Гол после действия",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            for row in selected_episodes.head(3).itertuples():
                st.link_button(
                    f"{row.date} · {row.opponent} · {row.time}",
                    f"https://github.com/hudl/open-data/blob/{SOURCE_COMMIT}/data/events/{row.match_id}.json",
                    key=f"proof_{key}_{row.Index}",
                )


def highlighted_metrics(player, keys):
    data = metric_table(player, keys)
    signals = manager_signals[manager_signals.player_id.eq(player.player_id)]
    colours = colour_table(player, keys, metric_peers, signals)
    data["Сигнал"] = colours.status
    return data.style.apply(lambda row: [STYLES[colours.iloc[row.name].colour]] * len(row), axis=1)


def show_episodes(person_id):
    data = episodes[episodes.player_id.eq(person_id)].copy()
    if not st.checkbox("Показать также остальные потери и фолы", key=f"all_{person_id}"):
        data = data[data.danger]
    data = data.sort_values(["goal", "provider_error", "shot_xg", "date"], ascending=False)
    if data.empty:
        st.caption("По выбранным условиям эпизодов нет.")
        return
    st.dataframe(
        data[["date", "opponent", "time", "kind", "shot_xg", "goal"]].rename(
            columns={
                "date": "Дата",
                "opponent": "Соперник",
                "time": "Минута",
                "kind": "Действие",
                "shot_xg": "xG удара",
                "goal": "Гол соперника",
            }
        ),
        hide_index=True,
        height=min(330, 38 + 35 * len(data)),
        width="stretch",
        column_config={
            "xG удара": st.column_config.NumberColumn(format="%.3f"),
            "Гол соперника": st.column_config.CheckboxColumn(),
        },
    )
    st.caption("После действия — временная связь. Низкий xG удара не доказывает ошибку вратаря.")
    with st.expander("Проверить конкретный эпизод"):
        choices = dict(
            zip(
                data.index.astype(int),
                data.apply(lambda r: f"{r.date} · {r.opponent} · {r.time} · {r.kind}", axis=1),
                strict=True,
            )
        )
        selected = st.selectbox("Эпизод", choices, format_func=choices.get, key=f"episode_{person_id}")
        row = data.loc[selected]
        st.write(f"**{row.date} · {row.opponent} · счёт {row.score} · {row.time}**")
        st.write(row.kind)
        goal_events = query(
            "SELECT event_id, event_index, type, team_id, outcome FROM events WHERE match_id=? "
            "AND ((type='Shot' AND outcome='Goal') OR type='Own Goal Against') ORDER BY event_index",
            (int(row.match_id),),
        )
        target = query(
            "SELECT event_index FROM events WHERE event_id=?",
            (row.shot_id if pd.notna(row.shot_id) else row.event_id,),
        )
        origin = query("SELECT event_index FROM events WHERE event_id=?", (row.event_id,))
        if len(target) and len(origin):

            def score_at(index):
                subset = goal_events[goal_events.event_index.le(index)]
                ours = (
                    (subset.type.eq("Shot") & subset.team_id.eq(TEAM_ID))
                    | (subset.type.eq("Own Goal Against") & subset.team_id.ne(TEAM_ID))
                ).sum()
                return f"{int(ours)}:{len(subset) - int(ours)}"

            st.write(
                f"Счёт: {score_at(int(origin.iloc[0].event_index) - 1)} → {score_at(int(target.iloc[0].event_index))} (Лестер:соперник)"
            )
        if pd.notna(row.shot_id):
            st.write(
                f"xG удара: {row.shot_xg:.3f} · гол: {'да' if row.goal else 'нет'}"
                + (f" · через {row.seconds_after:.1f} с" if pd.notna(row.seconds_after) else "")
            )
        if row.note:
            st.caption(row.note)
        st.code(f"Событие: {row.event_id}" + (f"\nУдар: {row.shot_id}" if pd.notna(row.shot_id) else ""))
        st.link_button(
            "Исходные данные матча",
            f"https://github.com/hudl/open-data/blob/{SOURCE_COMMIT}/data/events/{row.match_id}.json",
        )


def ai_summary(player):
    with st.expander("AI-пояснение по цифрам"):
        ready = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))
        if not ready:
            st.caption("Для генерации задайте OPENAI_API_KEY и OPENAI_MODEL в .env.")
        if st.button("Кратко объяснить сезон игрока", disabled=not ready, key=f"ai_{player.player_id}"):
            facts = [
                {"id": key, "fact": f"{LABELS[key]}: {player[key]}"}
                for key in LABELS
                if key in player and pd.notna(player[key])
            ]
            data = {
                "scope": "player-season",
                "player": player["name"],
                "role": player.role,
                "season": "2015/16",
                "team": "Leicester City",
                "evidence": facts,
            }
            try:
                with st.spinner("Формирую краткий разбор…"):
                    report = llm_report(data)
                for section in report["sections"]:
                    st.write(section["interpretation"])
                    st.caption("Факты: " + ", ".join(section["evidence_ids"]))
            except (APIError, ValueError, KeyError, TypeError):
                st.error("Не удалось получить отчёт. Проверьте доступ к API и выбранную модель.")


if page == "Менеджеру":
    st.subheader("Кто помогает команде — и что разобрать")
    st.caption(
        "Зелёный — сильный показатель; жёлтый — зона внимания; красный — приоритет разбора. Сильные стороны и риски независимы."
    )
    focus = st.radio("Фокус", ["Все", "Сильные стороны", "Риски"], horizontal=True)
    minimum = st.checkbox("Не менее 450 минут", value=True, key="manager_minimum")
    table = manager_summary.copy()
    if minimum:
        table = table[table.minutes.ge(450)]
    if focus == "Сильные стороны":
        table = table[table.contribution_colour.eq("green")]
    elif focus == "Риски":
        table = table[table.risk_colour.isin(["red", "yellow"])]
    table["priority"] = table.risk_colour.map({"red": 3, "yellow": 2, "green": 1, "grey": 0})
    table = table.sort_values(
        ["priority", "minutes"] if focus == "Риски" else ["strength_count", "minutes"], ascending=False
    )
    overview = pd.DataFrame(
        {
            "Игрок": table.name,
            "Что получается": table.strength_brief,
            "Риск / причина": table.reason_brief,
            "Что сделать": table.action,
        }
    ).reset_index(drop=True)
    good_colours, risk_colours = table.contribution_colour.tolist(), table.risk_colour.tolist()
    st.dataframe(
        overview.style.apply(
            lambda row: ["", COLOURS[good_colours[row.name]], COLOURS[risk_colours[row.name]], ""], axis=1
        ),
        hide_index=True,
        width="stretch",
        height=min(520, 38 + 35 * len(table)),
        column_config={
            "Игрок": st.column_config.TextColumn(width=140),
            "Что получается": st.column_config.TextColumn(width=235),
            "Риск / причина": st.column_config.TextColumn(width=235),
            "Что сделать": st.column_config.TextColumn(width=290),
        },
    )
    st.caption(
        "Сильный показатель: верхние 25% сопоставимых ролей АПЛ, минимум 450 минут. Красный — повод открыть эпизод, а не ярлык «плохой игрок»."
    )
    if len(table):
        options = dict(zip(table.player_id.astype(int), table.name, strict=True))
        pid = st.selectbox(
            "Разобрать игрока",
            options,
            index=list(options).index(3961) if 3961 in options else 0,
            format_func=options.get,
        )
        coloured_signals(pid, f"manager_{pid}")
    st.download_button(
        "Скачать менеджерскую сводку CSV",
        table.drop(columns="priority").to_csv(index=False),
        "leicester_manager.csv",
        "text/csv",
    )
elif page == "Игрок":
    ordered = players.sort_values("minutes", ascending=False)
    options = dict(zip(ordered.player_id.astype(int), ordered.name, strict=True))
    selected_id = st.selectbox(
        "Игрок", options, index=list(options).index(3961) if 3961 in options else 0, format_func=options.get
    )
    p = players[players.player_id.eq(selected_id)].iloc[0]
    st.markdown(
        f'<div class="player-title">{escape(p["name"])}</div><div class="muted">{escape(p.role)}</div>',
        unsafe_allow_html=True,
    )
    header_player = p.copy()
    header_player["goal_involvements"] = p.goals + p.assists
    header_peers = metric_peers.assign(goal_involvements=metric_peers.goals + metric_peers.assists)
    header_styles = []
    for index, (column, label, value) in enumerate(
        zip(
            st.columns(4),
            [
                "Средняя оценка TL",
                "Матчи / в старте",
                "Минуты",
                "Сейвы" if p.role == ROLES[0] else "Голы + пасы",
            ],
            [
                number(p.average_rating, 2),
                f"{int(p.matches)} / {int(p.starts)}",
                number(p.minutes, 1),
                str(int(p.saves)) if p.role == ROLES[0] else f"{int(p.goals)} + {int(p.assists)}",
            ],
            strict=True,
        )
    ):
        metric_key = [
            "average_rating",
            "matches",
            "minutes",
            "saves" if p.role == ROLES[0] else "goal_involvements",
        ][index]
        colour, reason = classify_metric(header_player, metric_key, header_peers)
        with column.container(key=f"player_header_{index}"):
            st.metric(label, value, help=reason)
        header_styles.append(f'.st-key-player_header_{index} [data-testid="stMetric"]{{{STYLES[colour]}}}')
    st.markdown("<style>" + "".join(header_styles) + "</style>", unsafe_allow_html=True)
    st.caption(
        f"Расчётная TL · {int(p.rated_matches)} матчей с 30+ минутами · шкала 0–10."
        + (" Малая выборка: менее 450 минут." if not p.reliable else "")
        + (" Для вратаря — без PSxG." if p.role == ROLES[0] else "")
    )
    core = CORE[ROLES[3]] if p.rating_profile == "Атакующая полузащита" else CORE[p.role]
    st.subheader("Вывод для менеджера")
    coloured_signals(selected_id, f"player_{selected_id}", compact=True)
    st.caption("Полные значения и сравнение приведены ниже; цвет не заменяет контекст эпизода.")
    st.subheader("Основные показатели")
    st.caption(
        "🟢 Сильный показатель / нет негативного события · 🟡 внимание · 🔴 приоритет · 🔵 норма или контекст · ⚪ мало данных"
    )
    st.dataframe(
        highlighted_metrics(p, core),
        hide_index=True,
        width="stretch",
        height=320,
        column_config={
            "Показатель": st.column_config.TextColumn(width=160),
            "За сезон": st.column_config.TextColumn(width=72),
            "На 90 мин": st.column_config.TextColumn(width=72),
            "Сигнал": st.column_config.TextColumn(width=160),
        },
    )
    discipline = []
    for key, label in [
        ("yellow_cards", "ЖК"),
        ("red_cards", "КК"),
        ("fouls", "Фолы"),
        ("penalties_conceded", "Пенальти в свои"),
        ("penalties_missed", "Незабитые пенальти"),
    ]:
        colour, reason = classify_metric(
            p, key, metric_peers, manager_signals[manager_signals.player_id.eq(selected_id)]
        )
        discipline.append(
            f'<span title="{escape(reason, quote=True)}" style="{STYLES[colour]};padding:3px 7px;border-radius:5px">{label}: {int(p[key])}</span>'
        )
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:6px;font-size:12px">' + "".join(discipline) + "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Все показатели"):
        st.dataframe(highlighted_metrics(p, list(LABELS)), hide_index=True, width="stretch", height=530)
    with st.expander("Почему у показателя такой цвет"):
        explanations = colour_table(
            p, list(LABELS), metric_peers, manager_signals[manager_signals.player_id.eq(selected_id)]
        )
        explanations["Показатель"] = explanations.metric.map(LABELS)
        st.dataframe(
            explanations[["Показатель", "status", "reason"]].rename(
                columns={"status": "Сигнал", "reason": "Причина"}
            ),
            hide_index=True,
            width="stretch",
        )
    st.subheader("Важные эпизоды")
    show_episodes(selected_id)
    with st.expander("Оценки и статистика по матчам"):
        pg = games[games.player_id.eq(selected_id)].sort_values("date", ascending=False)
        st.dataframe(
            pg[
                [
                    "date",
                    "opponent",
                    "score",
                    "minutes",
                    "rating",
                    "goals",
                    "assists",
                    "fouls",
                    "yellow_cards",
                    "red_cards",
                ]
            ]
            .rename(
                columns={
                    "date": "Дата",
                    "opponent": "Соперник",
                    "score": "Лестер:соперник",
                    "minutes": "Минуты",
                    "rating": "TL",
                    **LABELS,
                }
            )
            .round(2),
            hide_index=True,
            width="stretch",
        )
        game_options = dict(
            zip(pg.match_id.astype(int), pg.apply(lambda r: f"{r.date} · {r.opponent}", axis=1), strict=True)
        )
        mid = st.selectbox("Матч для проверки оценки", game_options, format_func=game_options.get)
        row = pg[pg.match_id.eq(mid)].iloc[0]
        st.caption(
            f"База метрик: {number(row.metric_score, 2)} · бонус: {min(1, row.penalties_saved * 0.6 + row.penalties_won * 0.2):.2f} · вычеты: {row.deduction:.2f}"
        )
        reasons = pd.DataFrame(json.loads(row.deduction_detail))
        if len(reasons):
            st.dataframe(
                reasons.rename(columns={"reason": "Действие", "cost": "Вычет", "episode": "Эпизод"}),
                hide_index=True,
                width="stretch",
            )
    ai_summary(p)
elif page == "Схема и роли":
    tactics = tactical_data(DB.stat().st_mtime_ns)
    st.subheader("Расстановка → роль → показатели")
    st.caption(
        "Позиции сезона 2015/16. Выберите матч и игрока: показатели сравниваются с футболистами АПЛ той же роли."
    )
    mids = list(tactics["lineups"])
    mid = st.selectbox(
        "Стартовый состав матча",
        mids,
        index=mids.index(tactics["representative"]),
        format_func=lambda value: (
            f"{tactics['lineups'][value]['date']} · {tactics['lineups'][value]['opponent']} · {tactics['lineups'][value]['score']}"
        ),
    )
    selected = tactics["lineups"][mid]
    lineup = selected["lineup"]
    ids = [p["player_id"] for p in lineup]
    if st.session_state.get("tactical_player") not in ids:
        st.session_state.tactical_player = 3961 if 3961 in ids else ids[0]
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown(f"**{'–'.join(selected['formation'])} · атака ↑**")
        event = st.plotly_chart(
            formation_figure(lineup, st.session_state.tactical_player),
            key=f"formation_{mid}",
            on_select="rerun",
            selection_mode="points",
            config={"displayModeBar": False},
            width="stretch",
        )
        points = event.selection.points
        # Apply each newly clicked selection once; a later dropdown choice stays selected.
        signature = (mid, tuple(p.get("point_index") for p in points))
        if points and st.session_state.get("last_pitch_selection") != signature:
            clicked = int(points[-1]["customdata"][0])
            if clicked in ids:
                st.session_state.tactical_player = clicked
                st.session_state.last_pitch_selection = signature
                st.rerun()
        st.caption(
            "Стартовые позиции StatsBomb; расстояния условные. Нажмите на футболку или выберите игрока справа."
        )
    with right:
        pid = st.selectbox(
            "Игрок на схеме",
            ids,
            key="tactical_player",
            format_func=lambda value: NAMES.get(
                value, next(p["name"] for p in lineup if p["player_id"] == value)
            ),
        )
        person = next(p for p in lineup if p["player_id"] == pid)
        st.markdown(f"**№{person['number']} · {person['position_ru']}**")
        st.caption(
            f"Стартов за сезон: {tactics['starts'][pid]} · на этой позиции: {tactics['positional_starts'][pid].get(person['position'], 0)}"
        )
        player, colours, family = tactical_comparison(pid, person["position"], DB.stat().st_mtime_ns)
        st.write(ROLE_NOTES[family])
        if player is not None:
            st.caption(
                f"В этой группе ролей: {int(player.matches)} матчей · {number(player.minutes)} мин. Статистика ниже — только этих матчей."
            )
            data = metric_table(player, ROLE_KEYS[family]).rename(columns={"За сезон": "В этой роли"})
            data["Сигнал"] = colours.status
            st.dataframe(
                data.style.apply(lambda row: [STYLES[colours.iloc[row.name].colour]] * len(row), axis=1),
                hide_index=True,
                width="stretch",
            )
            strong = colours[colours.colour.eq("green") & ~colours.status.str.contains("Нет события")]
            review = colours[colours.colour.isin(["red", "yellow"])]
            if len(strong):
                st.success("Сильные показатели: " + "; ".join(LABELS[k] for k in strong.metric) + ".")
            if len(review):
                st.warning(
                    "Проверить по эпизодам: "
                    + "; ".join(LABELS[k] for k in review.metric)
                    + ". Это не доказательство вины в поражении."
                )
            with st.expander("Почему такие цвета и как учитывается роль"):
                st.dataframe(
                    colours.assign(Показатель=colours.metric.map(LABELS))[["Показатель", "status", "reason"]],
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Матч относится к группе по наиболее частой позиции игрока в событиях. При смене роли внутри матча минуты не разделяются. Сравнение — с игроками АПЛ с ≥450 минут в этой группе; счётчики на 90 минут. Роль определяет набор показателей, но не доказывает тренерское задание."
                )
        else:
            st.info("Для этой роли нет сопоставимых матчей игрока.")
    st.divider()
    a, b, c, d = st.columns(4)
    counter = tactics["patterns"]["From Counter"]
    a.metric("Стартовая схема 4–4–2", f"{tactics['formations'].get(442, 0)}/38")
    b.metric("Удары из контратак*", int(counter["shots"]))
    c.metric("Голы из контратак*", int(counter["goals"]))
    d.metric("xG этих ударов*", f"{counter['xg']:.2f}")
    st.caption(
        "Только удары с меткой StatsBomb «From Counter»; это не число всех контратак. xG — оценка StatsBomb."
    )
    st.markdown(
        f"Компактная оборона и быстрые переходы — описание стиля Раньери в [разборе Premier League]({TACTICS_SOURCE}). Там же: **42,43% владения**. В такой модели вклад опорника оцениваем прежде всего по возвратам мяча и защите зоны перед защитниками, флангового игрока — по созданию моментов."
    )
    with st.expander("Состав, источники и ограничения"):
        st.write(
            f"Самая частая группа из 11 стартующих игроков встречается в {tactics['common_xi_matches']} матчах. По умолчанию показан реальный матч с её наиболее частым распределением позиций."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Схема": "–".join(str(key)), "Матчей": value}
                    for key, value in tactics["formations"].items()
                ]
            ),
            hide_index=True,
        )
        st.markdown(
            f"[Подложка поля: Wikimedia Commons, public domain]({PITCH_SOURCE}) · [Исходные события StatsBomb](https://github.com/hudl/open-data/tree/{SOURCE_COMMIT}/data/events). Футболки и подписи добавлены приложением."
        )
        st.caption(
            "Это стартовая структура, не средние координаты и не схема каждой фазы игры. Официальный обзор объединяет 4–4–2 и 4–4–1–1; наши числа считают отдельные стартовые коды. Перемещения без мяча, персональные задания и физическая готовность по этим данным не определяются."
        )

elif page == "Состав":
    role_filter = st.radio("Позиция", ["Все", *ROLES], horizontal=True)
    reliable = st.checkbox("Не менее 450 минут", value=True)
    table = players.copy()
    if role_filter != "Все":
        table = table[table.role.eq(role_filter)]
    if reliable:
        table = table[table.reliable.eq(1)]
    st.dataframe(
        table.sort_values("average_rating", ascending=False)[
            [
                "name",
                "role",
                "matches",
                "minutes",
                "average_rating",
                "goals",
                "assists",
                "xa",
                "recoveries",
                "penalties_conceded",
                "red_cards",
            ]
        ]
        .rename(
            columns={
                "name": "Игрок",
                "role": "Роль",
                "matches": "Матчи",
                "minutes": "Минуты",
                "average_rating": "Средняя TL",
                **LABELS,
            }
        )
        .round(2),
        hide_index=True,
        width="stretch",
        height=550,
    )
    st.subheader("Кто был ключевым")
    records = []
    for feature, label in [
        ("goals", "Голы"),
        ("assists", "Голевые передачи"),
        ("xa", "Созданные моменты · xA"),
        ("recoveries", "Возвраты мяча"),
        ("successful_interceptions", "Перехваты"),
        ("tackles_won", "Успешные отборы"),
    ]:
        eligible = players[players.reliable.eq(1)]
        leader = eligible.loc[eligible[feature].idxmax()]
        records.append(
            {
                "Показатель": label,
                "Игрок": leader["name"],
                "За сезон": number(leader[feature], 2 if feature == "xa" else 0),
            }
        )
    st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
    st.download_button("Скачать состав CSV", players.to_csv(index=False), "leicester_players.csv", "text/csv")
elif page == "Матчи":
    outcome = st.radio("Результат", ["Все", "Победы", "Ничьи", "Поражения"], horizontal=True)
    table = (
        matches
        if outcome == "Все"
        else matches[matches.result.eq({"Победы": "В", "Ничьи": "Н", "Поражения": "П"}[outcome])]
    )
    st.dataframe(
        table[["date", "opponent", "venue", "score", "result"]].rename(
            columns={
                "date": "Дата",
                "opponent": "Соперник",
                "venue": "Поле",
                "score": "Лестер:соперник",
                "result": "Исход",
            }
        ),
        hide_index=True,
        width="stretch",
        height=350,
    )
    choices = dict(
        zip(
            table.match_id.astype(int),
            table.apply(lambda r: f"{r.date} · {r.opponent} · {r.score}", axis=1),
            strict=True,
        )
    )
    mid = st.selectbox("Матч", choices, format_func=choices.get)
    rows = games[games.match_id.eq(mid)].copy()
    rows["name"] = rows.player_id.map(NAMES).fillna(rows.player)
    st.dataframe(
        rows.sort_values("rating", ascending=False)[
            [
                "name",
                "role",
                "minutes",
                "rating",
                "goals",
                "assists",
                "penalties_conceded",
                "yellow_cards",
                "red_cards",
            ]
        ]
        .rename(columns={"name": "Игрок", "role": "Роль", "minutes": "Минуты", "rating": "TL", **LABELS})
        .round(2),
        hide_index=True,
        width="stretch",
    )
else:
    st.subheader("Цветовые сигналы")
    st.write(
        "Зелёный: показатель на 90 минут в верхней четверти сезонных значений игроков той же роли/профиля АПЛ, при 450+ минутах. "
        "Жёлтый: низкая точность передач, частые потери в своей половине, недобор реализации, неудачные выходы или эпизоды для проверки. "
        "Красный: пенальти в свои ворота, удаление, автогол, два незабитых пенальти, Error перед голевым ударом или повторные цепочки перед голевыми ударами. "
        "Это правила приоритизации разбора, не причинная оценка вины и не решение о составе."
    )
    st.subheader("Как считается оценка")
    st.write(
        "TL — наш описательный индекс, не официальный рейтинг. База 6 — середина показателей сопоставимых игроков АПЛ 2015/16."
    )
    st.code(
        "Оценка = 6 + 6 × (взвешенный процентиль − 0,5) + бонусы − вычеты\nСредняя за сезон = сумма(оценка × минуты) / сумма минут оценённых матчей"
    )
    st.write(
        "Оценка выставляется при 30+ минутах. Для сравнения используются матчи игроков с 450+ минутами за сезон. Счётчики приведены к 90 минутам. Атакующая полузащита имеет отдельный профиль."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Профиль": name, "Показатель": LABELS.get(feature, feature), "Вес": f"{weight:.0%}"}
                for name, features in WEIGHTS.items()
                for feature, weight in features.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.write(
        "Вычеты: пенальти в свои / удаление −0,80; автогол −0,70; действие перед голевым ударом −0,50; незабитый пенальти −0,40; явный Error / неудачный выход −0,35; жёлтая −0,10. Один эпизод — максимальный вычет. Это выбранные нами веса, не доказанная стоимость ошибки."
    )
    st.write(
        "Малые вычеты: плохой приём 0,04; обыгрыш соперником 0,035; потеря после отбора 0,025; неудачная обводка 0,02; неточный пас 0,005; фол 0,015, у штрафной 0,06. Перед неголевым ударом: 0,15 × xG. Бонус: отражённый пенальти +0,60, заработанный +0,20; максимум +1 за матч. Итог ограничен 0–10."
    )
    st.subheader("Определения")
    st.write(
        "Минуты — в пределах 90 минут, без добавленного времени. Фол у своей штрафной: x≤30, 15≤y≤65 на сетке 120×80. Передача вперёд: точный пас с приростом x≥10. Офсайды — передачи, адресованные игроку с отметкой Pass Offside."
    )
    st.write(
        "Связь потери с ударом: 20 секунд без остановки и возврата контроля. После фола: одно владение со стандарта/преимущества, до 180 секунд. Это временная связь, не доказательство вины."
    )
    st.write(
        "Низкий xG не означает лёгкий сейв. PSxG отсутствует: голы с xG≤0,10 показаны для просмотра, сами по себе оценку не снижают. Без трекинга и видео нельзя установить, кому игрок должен был отдать невыбранную передачу."
    )
    with st.expander("ML-исследование и воспроизведение"):
        if (ARTIFACTS / "metrics.json").exists():
            ml = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))["xg"]
            st.dataframe(
                pd.DataFrame(
                    [{"Модель": key, **value["test"]} for key, value in ml["candidates"].items()]
                ).round(4),
                hide_index=True,
            )
        st.code(
            "python -m touchline.download\npython -m touchline.ingest\npython -m touchline.modeling\npython -m touchline.research\npython -m touchline.season\nstreamlit run app.py"
        )
    st.caption(f"Data source: StatsBomb Open Data · версия {SOURCE_COMMIT[:10]}")
    st.image(str(ROOT / "assets/statsbomb-logo.png"), width=160)
    st.link_button("Источник данных", "https://github.com/hudl/open-data")
