"""Ground reports in inspectable metrics and original event IDs."""

import json
import os

from touchline.config import ARTIFACTS


def packet(match, team_rows, shots, events, players=None, incidents=None):
    commit = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))["commit"]
    evidence = [
        {
            "id": "result",
            "fact": f"{match['home']} {match['home_score']}–{match['away_score']} {match['away']}",
        }
    ]
    for row in team_rows.to_dict("records"):
        for key, label in [
            ("xg", "StatsBomb xG"),
            ("shots", "удары"),
            ("box_entries", "входы в штрафную"),
            ("pressures", "зарегистрированные действия давления"),
            ("red_cards", "удаления"),
        ]:
            value = row[key]
            text = f"{value:.2f}" if key == "xg" else str(int(value))
            evidence.append(
                {"id": f"team_{row['team_id']}_{key}", "fact": f"{row['team']}: {label} = {text}"}
            )
    # Goals and the most valuable chances, with original event UUIDs.
    selected = shots.sort_values(["goal", "statsbomb_xg"], ascending=False).head(12)
    names = dict(zip(team_rows.team_id, team_rows.team, strict=False))
    for shot in selected.to_dict("records"):
        evidence.append(
            {
                "id": shot["event_id"],
                "fact": (
                    f"{shot['minute']}' · {names.get(shot['team_id'])} · {shot['player']} · "
                    f"{'гол' if shot['goal'] else 'незабитый удар'} · StatsBomb xG {shot['statsbomb_xg']:.3f}"
                ),
            }
        )
    cards = events[events.card.isin(["Red Card", "Second Yellow"])]
    for own_goal in events[events.type.eq("Own Goal Against")].to_dict("records"):
        beneficiary = match["away"] if own_goal["team_id"] == match["home_id"] else match["home"]
        evidence.append(
            {
                "id": own_goal["event_id"],
                "fact": (
                    f"{own_goal['minute']}' · {own_goal['player']} · автогол команды {own_goal['team']} "
                    f"в пользу {beneficiary}"
                ),
            }
        )
    for card in cards.to_dict("records"):
        evidence.append(
            {
                "id": card["event_id"],
                "fact": f"{card['minute']}' · {card['team']} · {card['player']} · {card['card']}",
            }
        )
    if players is not None:
        for p in players.to_dict("records"):
            evidence.append(
                {
                    "id": f"player_{p['player_id']}",
                    "fact": (
                        f"{p['team']} · {p['role']} · {p['player']}: голы {p['goals']}, xG {p['xg']:.3f}, "
                        f"пасы под удар {p['key_passes']}, xA {p['xa']:.3f}, сейвы {p['saves']}, "
                        f"пропущено в воротах {p['goals_conceded']}, успешные отборы {p['tackles_won']}/{p['tackles']}, "
                        f"плохая обработка {p['miscontrols']}, мяч отобрали {p['dispossessed']}, "
                        f"явные Error {p['provider_errors']}. После действий (временная связь): "
                        f"удары {p['shots_after']}, голы {p['goals_after']}."
                    ),
                }
            )
    if incidents is not None:
        selected_incidents = incidents[incidents.primary_link.eq(True) | incidents.provider_error.eq(True)]
        selected_incidents = selected_incidents.sort_values(
            ["provider_error", "shot_goal", "shot_xg"], ascending=False
        ).head(12)
        for action in selected_incidents.to_dict("records"):
            fact = f"{action['minute']}:{action['second']:02d} · {action['player']} ({action['team']}) · {action['kind']}."
            if action["shot_id"]:
                fact += (
                    f" В эпизоде соперника через {action['seconds_after']:.1f} с — удар "
                    f"{action['shot_player']}, xG {action['shot_xg']:.3f}, "
                    f"{'гол' if action['shot_goal'] else 'без гола'}, UUID {action['shot_id']}. "
                    "Временная связь не доказывает личную вину."
                )
            evidence.append({"id": action["event_id"], "fact": fact})
    return {
        "match_id": int(match["match_id"]),
        "date": match["date"],
        "evidence": evidence,
        "source": f"https://github.com/hudl/open-data/blob/{commit}/data/events/{match['match_id']}.json",
        "scope": "Post-match observational analysis; no tracking, injury or video data.",
    }


def deterministic_report(data):
    facts = data["evidence"]
    xg = [item for item in facts if item["id"].endswith("_xg")]
    entries = [item for item in facts if item["id"].endswith("_box_entries")]
    report = {
        "title": facts[0]["fact"],
        "sections": [
            {
                "heading": "Качество моментов",
                "interpretation": "Сравните суммарное качество ударов с итоговым счётом. "
                "xG описывает созданные моменты; преимущество по xG не гарантирует победу.",
                "evidence_ids": ["result", *[item["id"] for item in xg]],
            },
            {
                "heading": "Продвижение мяча",
                "interpretation": "Входы в штрафную показывают, как часто команда "
                "доставляла мяч в опасную зону передачей или ведением. Их количество не равно качеству моментов.",
                "evidence_ids": [item["id"] for item in entries],
            },
        ],
        "limitations": [
            "Это автоматический статистический разбор без вызова LLM.",
            "Наблюдаемые связи не доказывают причинность.",
            "Нет данных трекинга, травм и полного тактического контекста.",
        ],
    }
    player_facts = [item["id"] for item in facts if item["id"].startswith("player_")]
    if player_facts:
        report["sections"].append(
            {
                "heading": "Игроки и линии команды",
                "interpretation": "Ниже приведены индивидуальные показатели вратарей, защитников, полузащитников и нападающих. "
                "Отдельно учитываются явные Error провайдера, технические неудачи и последующие удары. "
                "Неудачное действие само по себе не доказывает, кто виноват в результате матча.",
                "evidence_ids": player_facts,
            }
        )
    return report


REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "heading": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["heading", "interpretation", "evidence_ids"],
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "sections", "limitations"],
}


def validate_report(report, data):
    allowed = {item["id"] for item in data["evidence"]}
    if not report.get("sections") or not report.get("limitations"):
        raise ValueError("Report must include analysis and limitations")
    for section in report["sections"]:
        ids = section.get("evidence_ids", [])
        if not ids or not set(ids).issubset(allowed):
            raise ValueError("Report contains an unknown or missing evidence reference")
    return report


def llm_report(data):
    from openai import OpenAI

    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not os.environ.get("OPENAI_API_KEY") or not model:
        raise ValueError("Укажите OPENAI_API_KEY и OPENAI_MODEL в .env")
    client = OpenAI(timeout=60, max_retries=1)
    response = client.responses.create(
        model=model,
        store=False,
        instructions=(
            "По-русски объясни сезон указанного игрока Leicester City в трёх коротких разделах, "
            "не более одного предложения в каждом: вклад, неудачные действия, ограничение оценки. "
            "Используй только evidence и существующие evidence_ids; не выдумывай значения. "
            "Это сезонные суммы, не один матч. Отсутствие ключевого паса не доказывает упущенную передачу. "
            "Обычный xG не определяет сложность сейва. Не приписывай вину в поражении по временной связи. "
            "Низкий xG пропущенного удара не является подтверждённой ошибкой вратаря."
        )
        if data.get("scope") == "player-season"
        else (
            "Ты футбольный аналитик. Напиши по-русски разбор сыгранного матча: "
            "отдельно вратари, защитники, полузащитники, нападающие обеих команд и важные эпизоды. "
            "Опиши, что получилось, что не получилось и у кого какие подтверждённые неудачные действия. "
            "Используй только переданный evidence. Каждый раздел обязан ссылаться "
            "на существующие evidence_ids. Все количественные утверждения должны точно "
            "совпадать с фактами. Не придумывай тактику, травмы, владение мячом или действия без "
            "подтверждения. Отделяй наблюдение от гипотезы. xG не является гарантированным счётом. "
            "Сумма xG не доказывает, кто заслужил победу. Не заявляй причинность. "
            "Явная ошибка Error провайдера отличается от неточного паса или проигранного единоборства. "
            "После действия не означает из-за действия. Не называй игрока виновником поражения "
            "по одной временной связи. Ноль передач под удар не доказывает, что игрок не отдал "
            "доступный ключевой пас. Нет PSxG: обычный xG не даёт оценки ошибок вратаря при сейвах. "
            "Укажи ограничения, включая отсутствие трекинга и наблюдательный характер данных."
        ),
        input=json.dumps(data, ensure_ascii=False),
        text={
            "format": {"type": "json_schema", "name": "match_report", "strict": True, "schema": REPORT_SCHEMA}
        },
    )
    return validate_report(json.loads(response.output_text), data)


def markdown_report(report, data):
    lookup = {item["id"]: item["fact"] for item in data["evidence"]}
    lines = [f"# {report['title']}", "", f"Матч: {data['match_id']} · {data['date']}", ""]
    for section in report["sections"]:
        lines.extend([f"## {section['heading']}", "", section["interpretation"], ""])
        lines.extend(f"- {lookup[key]} `[evidence: {key}]`" for key in section["evidence_ids"])
        lines.append("")
    lines.extend(
        [
            "## Ограничения",
            "",
            *[f"- {s}" for s in report["limitations"]],
            "",
            f"[Исходные события StatsBomb]({data['source']})",
        ]
    )
    return "\n".join(lines)
