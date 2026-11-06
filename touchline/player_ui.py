"""Player review interface; facts and temporal associations are labelled separately."""

import pandas as pd
import streamlit as st

from touchline import charts
from touchline.players import ROLES, role_findings

DISPLAY = {
    "player": "Игрок",
    "position": "Позиция",
    "minutes": "Минуты",
    "passes": "Передачи",
    "completed_passes": "Точные",
    "completion_pct": "Точность, %",
    "progressive_passes": "Вперёд ≥10 ед.",
    "key_passes": "Пасы под удар",
    "xa": "xA",
    "assists": "Голевые пасы",
    "shots": "Удары",
    "goals": "Голы",
    "xg": "xG",
    "miscontrols": "Плохая обработка",
    "dispossessed": "Мяч отобрали",
    "unsuccessful_passes": "Незавершённые пасы",
    "provider_errors": "Error от StatsBomb",
    "shots_after": "После: удары",
    "goals_after": "После: голы",
    "xg_after": "После: xG",
    "saves": "События сейвов",
    "goals_conceded": "Пропущено",
    "tackles": "Попытки отбора",
    "tackles_won": "Успешные отборы",
    "interceptions": "События перехвата",
    "blocks": "Блокировки",
    "clearances": "Выносы",
    "dribbled_past": "Обыгран",
    "pressures": "Давление",
}
ROLE_COLUMNS = {
    ROLES[0]: [
        "player",
        "minutes",
        "saves",
        "goals_conceded",
        "completion_pct",
        "unsuccessful_passes",
        "provider_errors",
        "shots_after",
        "goals_after",
    ],
    ROLES[1]: [
        "player",
        "minutes",
        "tackles_won",
        "tackles",
        "interceptions",
        "blocks",
        "clearances",
        "dribbled_past",
        "miscontrols",
        "provider_errors",
        "shots_after",
        "goals_after",
    ],
    ROLES[2]: [
        "player",
        "minutes",
        "passes",
        "completion_pct",
        "progressive_passes",
        "key_passes",
        "xa",
        "miscontrols",
        "dispossessed",
        "unsuccessful_passes",
        "shots_after",
        "goals_after",
    ],
    ROLES[3]: [
        "player",
        "minutes",
        "goals",
        "shots",
        "xg",
        "key_passes",
        "xa",
        "assists",
        "miscontrols",
        "dispossessed",
        "shots_after",
        "goals_after",
    ],
}


def render_player_review(players, incidents, events, shots, team, window):
    selected = players[players.team.eq(team)]
    team_id = int(selected.team_id.iloc[0])
    relevant = incidents[incidents.team_id.eq(team_id)].copy()
    context = f"{int(events.match_id.iloc[0])}_{team_id}"
    st.subheader("Что пошло не так в конкретных эпизодах")
    priorities = relevant[
        relevant.provider_error.eq(True) | (relevant.primary_link.eq(True) & relevant.shot_goal.eq(True))
    ]
    priorities = priorities.sort_values(["shot_goal", "provider_error", "shot_xg"], ascending=False)
    if priorities.empty:
        st.write(
            "В выбранном окне не найдено отметок Error или выделенных действий перед голевым ударом. "
            "Ниже доступны другие потери, неудачные обработки и передачи. Это не означает, что команда сыграла без ошибок."
        )
    for item in priorities.head(5).itertuples():
        text = f"**{item.minute}:{item.second:02d} · {item.player}: {item.kind.lower()}.**"
        if pd.notna(item.shot_id):
            text += (
                f" Через {item.seconds_after:.1f} с {item.shot_player} нанёс удар (xG {item.shot_xg:.3f})"
                + (", завершившийся голом." if item.shot_goal else "; гола не было.")
            )
        st.markdown(text)
    st.caption(
        "Это подтверждённые действия и последовательность событий. Последующее завершение атаки "
        "зависело и от других игроков; одна такая цепочка не определяет единственного виновника поражения."
    )
    st.subheader("Как сыграла каждая линия")
    st.caption(
        "Показатели с учётом роли игрока. Откройте эпизод ниже, чтобы проверить контекст неудачного действия."
    )
    for column, name in zip(st.columns(4), ROLES, strict=True):
        group = selected[selected.role.eq(name)]
        column.markdown(f"**{name}**")
        if group.empty:
            column.caption("Нет размеченных игроков")
            continue
        if name == ROLES[0]:
            column.metric(
                "Сейвы / пропущено", f"{int(group.saves.sum())} / {int(group.goals_conceded.sum())}"
            )
        elif name == ROLES[1]:
            column.metric(
                "Успешные отборы / попытки", f"{int(group.tackles_won.sum())} / {int(group.tackles.sum())}"
            )
        elif name == ROLES[2]:
            column.metric("Пасы под удар / xA", f"{int(group.key_passes.sum())} / {group.xa.sum():.2f}")
        else:
            column.metric("Голы / xG", f"{int(group.goals.sum())} / {group.xg.sum():.2f}")
    for role_tab, name in zip(st.tabs(ROLES), ROLES, strict=True):
        with role_tab:
            group = selected[selected.role.eq(name)]
            for finding in role_findings(group):
                st.write(finding)
            st.dataframe(
                group[ROLE_COLUMNS[name]].rename(columns=DISPLAY).round(2), hide_index=True, width="stretch"
            )
    st.caption(
        "Роль — наиболее частая позиция в разметке действий; вингеры включены в нападение. "
        "Минуты рассчитаны по стартовому составу, заменам и удалениям, включая добавленное время; "
        "временные уходы за поле не вычтены. Перехваты и сейвы — числа событий, не универсальная оценка качества."
    )
    with st.expander("Самые качественные незабитые удары команды"):
        missed = (
            shots[shots.team_id.eq(team_id) & shots.goal.eq(0)]
            .sort_values("statsbomb_xg", ascending=False)
            .head(5)
        )
        st.dataframe(
            missed[["minute", "player", "statsbomb_xg", "event_id"]].rename(
                columns={"minute": "Минута", "player": "Игрок", "statsbomb_xg": "xG удара"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Показаны созданные возможности, не завершившиеся голом. Сейв соперника, блокировка и промах "
            "различаются; высокий xG сам по себе не доказывает ошибку нападающего."
        )
    st.subheader("Какие эпизоды стоит проверить")
    names = ["Все игроки", *sorted(selected.player.unique())]
    chosen = st.selectbox("Игрок для разбора", names, key=f"review_player_{context}")
    if chosen != "Все игроки":
        relevant = relevant[relevant.player.eq(chosen)]
        person = selected[selected.player.eq(chosen)]
        st.markdown(f"**{chosen} · {person.role.iloc[0]}**")
        for finding in role_findings(person):
            st.write(finding)
    categories = sorted(relevant.kind.unique())
    filters = st.expander("Фильтры по типу действия").multiselect(
        "Тип действия", categories, default=categories, key=f"review_kind_{context}_{chosen}"
    )
    relevant = relevant[relevant.kind.isin(filters)]
    dangerous = st.checkbox(
        "Только отметки Error и эпизоды с последующим ударом", value=True, key=f"review_dangerous_{context}"
    )
    if dangerous:
        relevant = relevant[relevant.shot_id.notna() | relevant.provider_error.eq(True)]
    relevant = relevant.sort_values(
        ["provider_error", "shot_goal", "shot_xg", "event_index"], ascending=[False, False, False, True]
    )
    st.expander("Как определяется последующий удар").caption(
        f"Связываем действие с первым ударом соперника в пределах {window} секунд, в том же тайме "
        "и одном владении соперника, без возврата контроля и остановки игры. Несколько действий "
        "могут находиться в цепочке одного удара. В итогах по игрокам такой удар учитывается один раз, "
        "у последнего действия команды. Это правило поиска эпизодов, а не доказательство вины."
    )
    if relevant.empty:
        st.info("По выбранным фильтрам эпизодов нет. Это не доказывает, что игрок сыграл без ошибок.")
        return
    table = relevant[
        [
            "minute",
            "second",
            "player",
            "kind",
            "under_pressure",
            "recipient",
            "shot_xg",
            "shot_goal",
            "seconds_after",
            "primary_link",
        ]
    ].copy()
    table["time"] = table.minute.astype(str) + ":" + table.second.map(lambda x: f"{x:02d}")
    st.expander("Таблица найденных эпизодов").dataframe(
        table.drop(columns=["minute", "second"]).rename(
            columns={
                "time": "Время",
                "player": "Игрок",
                "kind": "Действие",
                "under_pressure": "Под давлением",
                "recipient": "Адресат передачи",
                "shot_xg": "xG следующего удара",
                "shot_goal": "Удар завершился голом",
                "seconds_after": "Секунд до удара",
                "primary_link": "Учтён в итогах",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    options = dict(
        zip(
            relevant.event_id,
            relevant.apply(
                lambda r: (
                    f"{r.minute}:{r.second:02d} · {r.player} · {r.kind}"
                    + (
                        f" → {'гол' if r.shot_goal else 'удар'} (xG {r.shot_xg:.2f})"
                        if pd.notna(r.shot_id)
                        else ""
                    )
                ),
                axis=1,
            ),
            strict=True,
        )
    )
    incident_id = st.selectbox(
        "Открыть эпизод", options, format_func=options.get, key=f"review_episode_{context}_{chosen}_{window}"
    )
    episode = relevant[relevant.event_id.eq(incident_id)].iloc[0]
    st.markdown(f"**{options[incident_id]}**")
    st.write(episode.observation)
    if pd.notna(episode.shot_id):
        st.write(
            f"Через {episode.seconds_after:.1f} с удар нанёс {episode.shot_player}; "
            f"xG {episode.shot_xg:.3f}. "
            + ("Этот удар завершился голом." if episode.shot_goal else "Гола после этого удара не было.")
        )
    st.caption("Под давлением соперника" if episode.under_pressure else "Давление не отмечено в разметке")
    st.plotly_chart(
        charts.incident_map(events, team_id, int(episode.event_index), int(episode.end_index)),
        width="stretch",
        key="incident_pitch",
    )
    st.caption(
        "Выбранная команда атакует вправо, соперник — влево. Линии показывают действия с мячом; "
        "это последовательность зарегистрированных координат, а не видеозапись или трекинг всех игроков."
    )
    chain = events[events.event_index.between(episode.event_index, episode.end_index)]
    st.expander("Последовательность событий").dataframe(
        chain[["minute", "second", "team", "player", "type", "outcome", "event_id"]],
        hide_index=True,
        width="stretch",
    )
    st.code(
        f"Исходное действие: {incident_id}"
        + (f"\nУдар: {episode.shot_id}" if pd.notna(episode.shot_id) else "")
    )
    st.download_button(
        "Скачать действия игроков CSV",
        selected.rename(columns=DISPLAY).to_csv(index=False),
        file_name=f"players_{int(events.match_id.iloc[0])}.csv",
        mime="text/csv",
    )
    with st.expander("Почему нельзя автоматически сказать «не отдал ключевой пас»?"):
        st.write(
            "Видны выполненные передачи, их адресаты и передачи под удар. Отсутствие такого паса не "
            "доказывает, что был свободный партнёр или доступная линия передачи. Для оценки невыбранных "
            "решений нужны позиции всех игроков в этот момент и обычно видео. В АПЛ 2015/16 этих данных нет."
        )
