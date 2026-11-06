"""Run with: streamlit run app.py"""

import json
import os
from html import escape

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import APIError

from touchline import charts
from touchline.analytics import fixed_shot_outcomes, score_states, state_summary
from touchline.config import ARTIFACTS, DB, ROOT
from touchline.data import factor_frame, match_events, match_shots, matches
from touchline.download import read_raw
from touchline.evidence import deterministic_report, llm_report, markdown_report, packet
from touchline.modeling import LABELS, SHOT_FEATURES, explain_factors
from touchline.player_ui import render_player_review
from touchline.players import analyze_players

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Touchline Lab · Premier League", page_icon="⚽", layout="wide")
st.markdown(
    """<style>
.block-container{padding-top:4rem;max-width:1500px} h1{letter-spacing:-.055em;font-weight:750!important}
[data-testid="stMetric"]{background:#141e29;padding:18px;border-radius:14px;border:1px solid #223040}
[data-testid="stMetricLabel"]{color:#8da0b5} .eyebrow{color:#a3e635;font-size:12px;letter-spacing:.18em}
.scoreboard{padding:28px;background:linear-gradient(110deg,#192b25,#141e29);border:1px solid #304436;
border-radius:18px;margin-bottom:22px}.scoreline{font-size:34px;font-weight:750;letter-spacing:-.03em}
.muted{color:#8da0b5;font-size:14px}.note{border-left:3px solid #a3e635;padding:10px 16px;color:#cbd5e1}
button[kind="primary"]{color:#0b1118!important} [data-testid="stSidebar"]{border-right:1px solid #223040}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Читаю события и рассчитываю показатели…")
def season_data(version):
    return matches(), factor_frame()


@st.cache_data
def selected_data(match_id, version):
    return match_events(match_id), match_shots(match_id)


@st.cache_data
def player_data(match_id, version, window):
    events, shots = selected_data(match_id, version)
    return analyze_players(read_raw(f"events/{match_id}.json"), events, shots, window)


@st.cache_resource
def models(version):
    return joblib.load(ARTIFACTS / "xg.joblib"), joblib.load(ARTIFACTS / "factors.joblib")


st.sidebar.markdown("### ⚽ Touchline Lab")
st.sidebar.caption("PREMIER LEAGUE · 2015/16")
page = st.sidebar.radio("Раздел", ["Разбор матча", "Команды и сезон", "Лаборатория ML", "Об исследовании"])
st.sidebar.divider()
st.sidebar.caption("Данные: StatsBomb Open Data / Hudl")
st.sidebar.markdown("[Источник и условия использования](https://github.com/hudl/open-data)")
if not DB.exists():
    st.title("Touchline Lab")
    st.info("Сначала подготовьте данные. Команды запуска приведены ниже и в README.")
    st.code(
        "python -m touchline.download\npython -m touchline.ingest\npython -m touchline.modeling\npython -m touchline.research"
    )
    st.stop()

version = DB.stat().st_mtime_ns
match_df, frame = season_data(version)
trained = all((ARTIFACTS / filename).exists() for filename in ("metrics.json", "xg.joblib", "factors.joblib"))
metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8")) if trained else None
if trained and metrics["match_count"] != len(match_df):
    trained = False
    metrics = None
    st.warning("Состав базы изменился. Переобучите модели: python -m touchline.modeling")
if trained:
    xg_model, factor_model = models((ARTIFACTS / "metrics.json").stat().st_mtime_ns)
st.sidebar.caption(f"В базе: {len(match_df)} матчей · {frame.team.nunique()} команд")
st.markdown('<div class="eyebrow">FOOTBALL, EXPLAINED WITH EVIDENCE</div>', unsafe_allow_html=True)

if page == "Разбор матча":
    st.title("Кто как сыграл и где возникли проблемы")
    st.caption("Вратарь, защита, полузащита и нападение — с конкретными действиями и цепочками событий.")
    c1, result_column, c2 = st.columns([1, 1, 3])
    team_options = sorted(frame.team.unique())
    team = c1.selectbox(
        "Команда",
        team_options,
        index=team_options.index("Leicester City") if "Leicester City" in team_options else 0,
    )
    options = match_df[(match_df.home == team) | (match_df.away == team)].sort_values("date", ascending=False)
    result_filter = result_column.selectbox("Результат команды", ["Все", "Поражения", "Победы", "Ничьи"])
    if result_filter != "Все":
        code = {"Поражения": "L", "Победы": "W", "Ничьи": "D"}[result_filter]
        options = options[
            options.match_id.isin(frame.loc[frame.team.eq(team) & frame.result.eq(code), "match_id"])
        ]
    if options.empty:
        st.info("В загруженной выборке нет матчей с этим исходом.")
        st.stop()
    choices = {
        int(m.match_id): f"{m.date} · {m.home} {m.home_score}–{m.away_score} {m.away}"
        for m in options.itertuples()
    }
    match_id = c2.selectbox("Матч", choices, format_func=choices.get)
    match = match_df[match_df.match_id.eq(match_id)].iloc[0].to_dict()
    events, shots = selected_data(match_id, version)
    rows = frame[frame.match_id.eq(match_id)].sort_values("is_home", ascending=False)
    focus = rows[rows.team.eq(team)].iloc[0]
    st.markdown(
        f'<div class="scoreboard"><div class="muted">{match["date"]} · '
        f"{escape(match['stadium'] or '')} · Тур {match['match_week']}</div>"
        f'<div class="scoreline">{escape(match["home"])} &nbsp; '
        f"{match['home_score']} : {match['away_score']} &nbsp; {escape(match['away'])}</div>"
        '<div class="muted">Послематчевый анализ · обе команды атакуют вправо на картах</div></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric(f"xG · {team}", f"{focus.xg:.2f}", f"{focus.xg_diff:+.2f} к сопернику", delta_color="off")
    cols[1].metric("Удары", int(focus.shots), f"{int(focus.shots_diff):+d} к сопернику", delta_color="off")
    cols[2].metric(
        "Входы в штрафную",
        int(focus.box_entries),
        f"{int(focus.box_entries_diff):+d} к сопернику",
        delta_color="off",
    )
    cols[3].metric(
        "Голы − xG", f"{focus.goals - focus.xg:+.2f}", "Разница результата и ожидания", delta_color="off"
    )
    st.caption(
        "xG в основных графиках — оценка StatsBomb. Голы − xG не являются чистой оценкой мастерства или удачи."
    )
    window = st.select_slider("Окно поиска удара после действия, секунд", [10, 15, 20, 30], value=20)
    players, incidents = player_data(match_id, version, window)
    tabs = st.tabs(["Игроки и ошибки", "Моменты", "Продвижение мяча", "Факторы результата", "AI-отчёт"])
    with tabs[0]:
        render_player_review(players, incidents, events, shots, team, window)
    with tabs[1]:
        a, b = st.columns(2)
        a.subheader("Где возникала опасность")
        a.plotly_chart(
            charts.shot_map(shots, match["home_id"], match["home"], match["away"]), width="stretch"
        )
        a.caption("Размер — xG удара. Звезда — гол. Все координаты приведены к направлению атаки вправо.")
        b.subheader("Как накапливались моменты")
        b.plotly_chart(
            charts.xg_timeline(shots, match["home_id"], match["home"], match["away"]), width="stretch"
        )
        b.caption(
            "Линия отражает накопленный xG, а не непрерывное измерение инициативы. Автоголы не являются ударами."
        )
        _, goals = score_states(events, match["home_id"], match["away_id"])
        with st.expander("Голы и состояние счёта"):
            st.dataframe(
                goals[["minute", "player", "own_goal", "score", "event_id"]], hide_index=True, width="stretch"
            )
            st.dataframe(
                state_summary(events, int(focus.team_id), match["home_id"], match["away_id"]),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Удар, завершившийся голом, относится к состоянию до гола. Удары / 90 ненадёжны "
                "на коротких интервалах. Состояние счёта влияет на поведение команд; это одна из причин "
                "не трактовать итоговую статистику как причинное объяснение."
            )
        with st.expander("Что было бы при повторении тех же моментов?"):
            replay = fixed_shot_outcomes(
                shots.loc[shots.team_id.eq(match["home_id"]), "statsbomb_xg"],
                shots.loc[shots.team_id.eq(match["away_id"]), "statsbomb_xg"],
            )
            for col, label, key in zip(
                st.columns(3),
                [match["home"], "Ничья", match["away"]],
                ["home_win", "draw", "away_win"],
                strict=False,
            ):
                col.metric(label, f"{replay[key]:.0%}")
            st.caption(
                "Точный расчёт для независимых ударов с вероятностями StatsBomb xG. Фиксируем уже "
                "созданный набор моментов; не моделируем новые атаки, изменения тактики и автоголы. "
                "Это условный мысленный эксперимент, а не прогноз и не оценка «заслуженной» победы."
            )
        display = shots[["minute", "player", "team_id", "statsbomb_xg", "goal", "event_id"]].copy()
        display["team"] = display.team_id.map(dict(zip(rows.team_id, rows.team, strict=False)))
        if trained:
            valid_coordinates = shots.x.between(0, 120) & shots.y.between(0, 80)
            display["own_xg"] = float("nan")
            if valid_coordinates.any():
                display.loc[valid_coordinates, "own_xg"] = xg_model.predict_proba(
                    shots.loc[valid_coordinates, SHOT_FEATURES]
                )[:, 1]
        st.subheader("Реестр ударов")
        st.dataframe(
            display.drop(columns="team_id"),
            hide_index=True,
            width="stretch",
            column_config={
                "statsbomb_xg": st.column_config.NumberColumn("StatsBomb xG", format="%.3f"),
                "own_xg": st.column_config.NumberColumn("Наш xG", format="%.3f"),
            },
        )
        st.caption(
            "Собственный xG — учебная модель на ограниченной выборке. Независимые метрики доступны в лаборатории ML."
        )
    with tabs[2]:
        a, b = st.columns([2, 1])
        network_team = a.selectbox("Команда на карте", rows.team.tolist())
        minimum = b.slider("Минимум передач в связи", 1, 10, 4)
        network_id = int(rows[rows.team.eq(network_team)].iloc[0].team_id)
        st.plotly_chart(charts.pass_network(events, network_id, minimum), width="stretch")
        st.caption(
            "Первый тайм. Узлы — средние координаты начала точных передач, линии — связи между игроками. "
            "Это не реальные средние позиции игроков и не данные трекинга."
        )
        comparison = rows[["team", "passes", "completed_passes", "box_entries", "pressures", "red_cards"]]
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.caption(
            "Вход в штрафную: точная передача или ведение из-за её пределов внутрь зоны x ≥ 102, 18 ≤ y ≤ 62. "
            "Количество событий Pressure не измеряет успешность прессинга."
        )
    with tabs[3]:
        st.subheader("Какие признаки повлияли на оценку модели")
        st.info(
            "Модель классифицирует исход по уже сыгранному матчу. Это объяснение статистической оценки, "
            "а не предматчевый прогноз и не доказательство причин результата."
        )
        if trained:
            explanation = explain_factors(factor_model, focus)
            names = {"W": "Победа", "D": "Ничья", "L": "Поражение"}
            for col, (label, probability) in zip(
                st.columns(3), explanation["probabilities"].items(), strict=False
            ):
                col.metric(names[label], f"{probability:.0%}")
            st.plotly_chart(charts.factor_chart(explanation, LABELS), width="stretch")
            st.caption(
                f"Контраст: {names[explanation['predicted']]} относительно «{names[explanation['reference']]}». "
                f"Базовый вклад (intercept): {explanation['intercept']:.3f}. "
                "Показаны точные слагаемые линейной модели в логарифме отношения вероятностей."
            )
            split = (
                "train"
                if match["date"] <= metrics["cutoffs"]["train_end"]
                else ("validation" if match["date"] <= metrics["cutoffs"]["validation_end"] else "test")
            )
            st.caption(f"Этот матч относится к выборке {split}. Для train оценка не является независимой.")
        else:
            st.warning("Обучите модели командой python -m touchline.modeling")
    with tabs[4]:
        data = packet(match, rows, shots, events, players, incidents)
        report_key = f"report_players_v1_{match_id}_{window}"
        mode_key = f"report_players_mode_v1_{match_id}_{window}"
        if report_key not in st.session_state:
            st.session_state[report_key] = deterministic_report(data)
            st.session_state[mode_key] = "Статистический шаблон · LLM не вызывалась"
        ready = bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"))
        st.caption(
            "LLM получает рассчитанные показатели и ID событий. Факты можно проверить под каждым выводом."
        )
        if not ready:
            st.info(
                "Для LLM-отчёта задайте OPENAI_API_KEY и OPENAI_MODEL в .env. Ниже доступен статистический шаблон."
            )
        if st.button("Сформировать LLM-разбор", type="primary", disabled=not ready):
            try:
                with st.spinner("Сопоставляю факты и формирую разбор…"):
                    st.session_state[report_key] = llm_report(data)
                    st.session_state[mode_key] = "LLM-разбор · проверьте интерпретации по фактам"
            except (APIError, ValueError, KeyError, TypeError):
                st.error(
                    "Не удалось получить корректный отчёт. Проверьте доступ к API, выбранную модель и подключение."
                )
        report = st.session_state[report_key]
        st.caption(st.session_state[mode_key])
        lookup = {e["id"]: e["fact"] for e in data["evidence"]}
        for section in report["sections"]:
            st.subheader(section["heading"])
            st.write(section["interpretation"])
            with st.expander("Факты и ссылки на события"):
                for ref in section["evidence_ids"]:
                    st.write(lookup[ref])
                    st.caption(ref)
        for limitation in report["limitations"]:
            st.caption(f"• {limitation}")
        st.link_button("Исходные события матча", data["source"])
        st.download_button(
            "Скачать отчёт .md",
            markdown_report(report, data),
            file_name=f"match_{match_id}.md",
            mime="text/markdown",
        )

elif page == "Команды и сезон":
    st.title("Счёт и качество игры")
    st.caption("Кто создавал моменты, кто реализовывал их лучше ожиданий и как менялась форма.")
    league = (
        frame.groupby("team")
        .agg(
            played=("points", "size"),
            points=("points", "sum"),
            goals=("goals", "sum"),
            conceded=("conceded", "sum"),
            xg=("xg", "sum"),
            xga=("xg_opp", "sum"),
            shots=("shots", "sum"),
        )
        .reset_index()
    )
    league["goal_diff"] = league.goals - league.conceded
    league["xg_diff"] = league.xg - league.xga
    league["residual"] = league.goals - league.xg
    league = league.sort_values(["points", "goal_diff", "goals"], ascending=False)
    a, b = st.columns([1, 1])
    with a:
        st.subheader("Создано и допущено")
        fig = px.scatter(
            league,
            x="xga",
            y="xg",
            text="team",
            size="points",
            color="points",
            color_continuous_scale=["#21485a", "#a3e635"],
            hover_data=["points", "goals"],
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_xaxes(title="xG соперников · меньше лучше")
        fig.update_yaxes(title="Собственный xG · больше лучше")
        st.plotly_chart(charts.style(fig, height=520), width="stretch")
    with b:
        st.subheader("Таблица сезона")
        st.dataframe(
            league[["team", "played", "points", "goals", "conceded", "xg", "xga", "residual"]].round(2),
            hide_index=True,
            height=520,
            width="stretch",
        )
    team = st.selectbox("Динамика команды", sorted(frame.team.unique()))
    trend = frame[frame.team.eq(team)].sort_values(["date", "match_id"]).copy()
    trend["rolling_xg_diff"] = trend.xg_diff.rolling(5, min_periods=5).mean()
    trend["rolling_points"] = trend.points.rolling(5, min_periods=5).mean()
    fig = px.line(
        trend,
        x="date",
        y="rolling_xg_diff",
        markers=True,
        hover_data=["opponent", "result"],
        color_discrete_sequence=[charts.GREEN],
    )
    fig.update_yaxes(title="Средняя разница xG за последние 5 матчей")
    fig.add_hline(y=0, line_dash="dot", line_color=charts.MUTED)
    st.plotly_chart(charts.style(fig), width="stretch")
    st.caption(
        "Описательная скользящая средняя включает показанный матч. Она не используется как предматчевый признак."
    )

elif page == "Лаборатория ML":
    st.title("Модель должна выдерживать проверку")
    st.caption("Разделение по времени, базовые модели, калибровка и прозрачные ограничения.")
    if not trained:
        st.info("Запустите python -m touchline.modeling")
        st.stop()
    a, b, c = st.columns(3)
    a.metric("Матчи", metrics["match_count"])
    b.metric("Удары", metrics["shot_count"])
    c.metric("Матчи в test", metrics["split_matches"]["test"])
    st.write(
        f"Обучение: до {metrics['cutoffs']['train_end']}; валидация: до "
        f"{metrics['cutoffs']['validation_end']}; тест: оставшиеся более поздние даты."
    )
    st.caption(
        "Все удары и обе командные строки одного матча принадлежат одной выборке. "
        "Выбор модели и регуляризации выполняется по validation log loss."
    )
    st.subheader("Собственный xG: вероятность гола после удара")
    xg = metrics["xg"]
    records = [{"model": name, **value["test"]} for name, value in xg["candidates"].items()]
    records += [
        {"model": "Постоянная частота гола (train)", **xg["baseline"]},
        {"model": "StatsBomb xG · внешний ориентир", **xg["provider_reference"]},
    ]
    st.dataframe(pd.DataFrame(records).round(4), hide_index=True, width="stretch")
    st.caption(
        f"По validation выбрана {xg['selected']}. 95% интервал test log loss: "
        f"{xg['test_log_loss_ci95'][0]:.3f}–{xg['test_log_loss_ci95'][1]:.3f}. "
        "Бутстрэп по матчам, 1000 повторов. Меньше log loss и Brier — лучше."
    )
    fig = px.scatter(
        x=xg["calibration"]["predicted"],
        y=xg["calibration"]["observed"],
        color_discrete_sequence=[charts.GREEN],
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=charts.MUTED, dash="dot"))
    fig.update_xaxes(title="Средняя предсказанная вероятность", range=[0, 1])
    fig.update_yaxes(title="Доля голов", range=[0, 1])
    st.plotly_chart(charts.style(fig, "Калибровка на test · 8 групп равного размера"), width="stretch")
    st.caption(
        "Признаки: дистанция, угол ворот, часть тела, тип и техника удара, давление, удар в одно касание. "
        "Исход удара, счёт, конечная точка удара и StatsBomb xG не подаются в нашу модель."
    )
    st.subheader("Интерпретируемая модель результата")
    factor = metrics["factors"]
    st.dataframe(
        pd.DataFrame(
            [
                {"model": "Логистическая регрессия", **factor["test"]},
                {"model": "Частоты исходов (train)", **factor["baseline"]},
            ]
        ).round(4),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Это послематчевая классификация победа/ничья/поражение. Для multiclass Brier используется "
        "сумма квадратов ошибок по трём классам; для xG — обычный бинарный Brier."
    )
    with st.expander("Полный протокол эксперимента"):
        st.json(metrics)
    st.warning(
        "Один исторический сезон ограничивает обобщение. StatsBomb xG использует более богатую информацию "
        "и не является независимой моделью, обученной нами. Интервалы не учитывают всю зависимость "
        "между матчами одних и тех же команд. Данные 2015/16 не описывают текущую АПЛ."
    )

else:
    st.title("Исследование, которое можно воспроизвести")
    st.markdown("""
**Вопрос исследования:** какие наблюдаемые игровые факторы связаны с победой или поражением в АПЛ 2015/16?

**Данные:** событийная разметка StatsBomb Open Data. Каждый матч содержит передачи, удары,
ведения, давление и другие события с координатами. Список матчей и события закреплены за коммитом источника;
SHA-256 каждого скачанного файла записывается в manifest.

**Архитектура:** JSON → нормализация Python/pandas → SQLite и SQL-представления → признаки →
scikit-learn → интерактивные графики → LLM-отчёт с проверяемыми ссылками на факты.

**Почему это не доказательство причин:** команда, которая проигрывает, может больше атаковать;
поэтому итоговые удары и передачи зависят и от хода матча. Учитывайте состояние счёта,
уровень соперника, удаления и ограничения выборки при интерпретации.

**LLM:** получает небольшой пакет рассчитанных фактов, а не весь сезон. Структурированный ответ
проходит проверку ссылок. Это проверяет наличие источников, но не гарантирует правильность
каждой интерпретации. В интерфейсе доступен реестр подтверждений и экспорт отчёта.

**Что не измеряется:** реальное владение по трекингу, непрерывные позиции всех игроков, травмы,
нагрузка и намерения тренера. Shot freeze frames не равны непрерывному трекингу.

**Атрибуция:** Data source: StatsBomb. Данные используются согласно условиям исходного репозитория;
лицензия кода проекта не распространяется на исходные данные.
""")
    if (ROOT / "assets/statsbomb-logo.png").exists():
        st.image(str(ROOT / "assets/statsbomb-logo.png"), width=170)
    st.link_button("Документация и данные StatsBomb", "https://github.com/hudl/open-data")
