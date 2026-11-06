"""Football-native charts drawn from coordinates, with explicit definitions."""

import numpy as np
import plotly.graph_objects as go

GREEN = "#a3e635"
BLUE = "#38bdf8"
MUTED = "#8da0b5"
PALETTE = [GREEN, BLUE]


def style(fig, title=None, height=370):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Arial"),
        margin=dict(l=20, r=20, t=45, b=25),
        title=title,
        height=height,
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def pitch():
    fig = go.Figure()
    line = dict(color="#526273", width=1)
    for x0, y0, x1, y1 in [
        (0, 0, 120, 80),
        (0, 18, 18, 62),
        (102, 18, 120, 62),
        (0, 30, 6, 50),
        (114, 30, 120, 50),
    ]:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=line)
    fig.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=line)
    fig.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=line)
    fig.add_shape(type="line", x0=120, y0=36, x1=120, y1=44, line=dict(color="#f8fafc", width=5))
    fig.update_xaxes(range=[-2, 123], visible=False, constrain="domain")
    fig.update_yaxes(range=[82, -2], visible=False, scaleanchor="x", scaleratio=1)
    return style(fig, height=440)


def shot_map(shots, home_id, home, away):
    fig = pitch()
    for i, (team_id, name) in enumerate([(home_id, home), (None, away)]):
        df = shots[shots.team_id.eq(team_id)] if team_id else shots[shots.team_id.ne(home_id)]
        for goal in (0, 1):
            subset = df[df.goal.eq(goal)]
            fig.add_trace(
                go.Scatter(
                    x=subset.x,
                    y=subset.y,
                    mode="markers",
                    name=f"{name} · {'гол' if goal else 'удар'}",
                    marker=dict(
                        size=9 + np.sqrt(subset.statsbomb_xg) * 34,
                        color=PALETTE[i],
                        symbol="star" if goal else "circle",
                        opacity=1 if goal else 0.55,
                        line=dict(width=1, color="#e2e8f0"),
                    ),
                    customdata=subset[["player", "minute", "statsbomb_xg", "event_id"]],
                    hovertemplate="%{customdata[0]} · %{customdata[1]}'<br>xG %{customdata[2]:.3f}"
                    "<br>%{customdata[3]}<extra>%{fullData.name}</extra>",
                )
            )
    return fig


def xg_timeline(shots, home_id, home, away):
    fig = go.Figure()
    end = max(90, shots.clock.max() if len(shots) else 90)
    for i, (team_id, name) in enumerate([(home_id, home), (None, away)]):
        df = shots[shots.team_id.eq(team_id)] if team_id else shots[shots.team_id.ne(home_id)]
        df = df.sort_values(["period", "clock"]).copy()
        df["cumulative"] = df.statsbomb_xg.cumsum()
        fig.add_trace(
            go.Scatter(
                x=[0, *df.clock, end],
                y=[0, *df.cumulative, df.cumulative.iloc[-1] if len(df) else 0],
                name=name,
                line=dict(color=PALETTE[i], width=3, shape="hv"),
            )
        )
        goals = df[df.goal.eq(1)]
        fig.add_trace(
            go.Scatter(
                x=goals.clock,
                y=goals.cumulative,
                mode="markers",
                showlegend=False,
                marker=dict(symbol="star", size=16, color=PALETTE[i]),
                text=goals.player,
                hovertemplate="Гол · %{text}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Минуты игры, включая добавленное время каждого тайма")
    fig.update_yaxes(title="Накопленный StatsBomb xG", rangemode="tozero")
    return style(fig)


def pass_network(events, team_id, minimum=4, period=1):
    """First-half pass network; centroids are pass origins, not tracking positions."""
    passes = events[
        (events.type == "Pass")
        & events.complete.eq(1)
        & events.team_id.eq(team_id)
        & events.period.eq(period)
    ].dropna(subset=["recipient_id", "x", "y"])
    fig = pitch()
    positions = passes.groupby("player_id").agg(
        x=("x", "mean"), y=("y", "mean"), player=("player", "first"), count=("player", "size")
    )
    edges = passes.groupby(["player_id", "recipient_id"]).size()
    for (sender, receiver), count in edges.items():
        if count < minimum or sender not in positions.index or receiver not in positions.index:
            continue
        p, q = positions.loc[sender], positions.loc[receiver]
        fig.add_trace(
            go.Scatter(
                x=[p.x, q.x],
                y=[p.y, q.y],
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(163,230,53,.35)", width=min(8, count / 2)),
                hovertemplate=f"{p.player} → {q.player}: {count} передач<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=positions.x,
            y=positions.y,
            mode="markers+text",
            showlegend=False,
            text=positions.player.map(lambda name: name.split()[-1]),
            textposition="top center",
            marker=dict(
                size=12 + np.sqrt(positions["count"]) * 2, color=GREEN, line=dict(color="#0b1118", width=2)
            ),
            customdata=positions[["player", "count"]],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]} точных передач<extra></extra>",
        )
    )
    return fig


def factor_chart(explanation, labels):
    items = sorted(explanation["contributions"].items(), key=lambda kv: abs(kv[1]))
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items],
            y=[labels[k] for k, _ in items],
            orientation="h",
            marker_color=[GREEN if v > 0 else BLUE for _, v in items],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Вклад в логарифм отношения вероятностей")
    return style(fig)


def incident_map(events, team_id, start_index, end_index):
    """Replay ball-action coordinates in the incident team's attacking direction."""
    fig = pitch()
    chain = events[events.event_index.between(start_index, end_index)]
    chain = chain[
        chain.type.isin(
            ["Pass", "Carry", "Dribble", "Shot", "Miscontrol", "Dispossessed", "Error", "Dribbled Past"]
        )
    ].dropna(subset=["x", "y"])
    for number, e in enumerate(chain.itertuples(), 1):
        own = e.team_id == team_id
        x, y = (e.x, e.y) if own else (120 - e.x, 80 - e.y)
        color = GREEN if own else BLUE
        if e.type in {"Pass", "Carry"} and not np.isnan(e.end_x) and not np.isnan(e.end_y):
            ex, ey = (e.end_x, e.end_y) if own else (120 - e.end_x, 80 - e.end_y)
            fig.add_annotation(
                x=ex,
                y=ey,
                ax=x,
                ay=y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=color,
                opacity=0.65,
                text="",
            )
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[str(number)],
                textposition="top center",
                showlegend=False,
                marker=dict(
                    size=15 if number == 1 else 10,
                    color=color,
                    symbol="star" if e.type == "Shot" else "circle",
                ),
                hovertemplate=f"{number}. {e.minute}:{e.second:02d} · {e.player} · {e.type}<extra></extra>",
            )
        )
    return fig
