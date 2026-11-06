"""Generate a real, data-backed research report and publication-ready figures."""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "touchline-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from touchline.config import ARTIFACTS, REPORTS, directories
from touchline.data import factor_frame, matches, query


def markdown_table(records, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                f"{record[k]:.4f}" if isinstance(record[k], float) else str(record[k]) for k in fields
            )
            + " |"
        )
    return "\n".join(lines)


def audit():
    counts = query(
        "SELECT (SELECT COUNT(*) FROM matches) AS matches, "
        "(SELECT COUNT(*) FROM events) AS events, (SELECT COUNT(*) FROM shots) AS shots"
    ).iloc[0]
    # Source score must equal shot goals plus Own Goal Against, counted once.
    checks = query("""
        SELECT m.match_id, m.home_score, m.away_score,
          SUM((e.type='Shot' AND e.outcome='Goal' AND e.team_id=m.home_id)
              OR (e.type='Own Goal Against' AND e.team_id=m.away_id)) AS event_home,
          SUM((e.type='Shot' AND e.outcome='Goal' AND e.team_id=m.away_id)
              OR (e.type='Own Goal Against' AND e.team_id=m.home_id)) AS event_away
        FROM matches m JOIN events e USING(match_id) GROUP BY m.match_id
    """)
    mismatches = checks[(checks.home_score != checks.event_home) | (checks.away_score != checks.event_away)]
    integrity = {
        **{k: int(v) for k, v in counts.items()},
        "score_mismatches": int(len(mismatches)),
        "invalid_shot_xg": int(
            query("SELECT COUNT(*) AS n FROM shots WHERE statsbomb_xg < 0 OR statsbomb_xg > 1").iloc[0].n
        ),
        "invalid_shot_coordinates": int(
            query("SELECT COUNT(*) AS n FROM shots WHERE x < 0 OR x > 120 OR y < 0 OR y > 80").iloc[0].n
        ),
        "gross_coordinate_errors": int(
            query("SELECT COUNT(*) AS n FROM shots WHERE x < -1 OR x > 121 OR y < -1 OR y > 81").iloc[0].n
        ),
        "own_goals": int(query("SELECT COUNT(*) AS n FROM events WHERE type='Own Goal Against'").iloc[0].n),
    }
    integrity["excluded_own_xg_shots"] = integrity["invalid_shot_coordinates"]
    if integrity["score_mismatches"] or integrity["invalid_shot_xg"] or integrity["gross_coordinate_errors"]:
        raise ValueError(f"Data quality checks failed: {integrity}")
    return integrity


def main():
    directories()
    figures = REPORTS / "figures"
    figures.mkdir(exist_ok=True)
    quality = audit()
    annotation_counts = (
        query(
            "SELECT type, COUNT(*) AS n FROM events WHERE type IN "
            "('Error','Miscontrol','Dispossessed') GROUP BY type"
        )
        .set_index("type")
        .n.to_dict()
    )
    (ARTIFACTS / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    m, frame = matches(), factor_frame()
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    league = frame.groupby("team").agg(
        points=("points", "sum"), goals=("goals", "sum"), xg=("xg", "sum"), xga=("xg_opp", "sum")
    )
    league["residual"] = league.goals - league.xg
    league.to_csv(REPORTS / "team_summary.csv")
    shots = pd.read_csv(ARTIFACTS / "shot_predictions.csv")
    test_shots = shots[shots.split.eq("test")]
    winners = frame[frame.result.eq("W")]
    xg_leaders = frame[frame.xg_diff > 0]
    home_win = float((m.home_score > m.away_score).mean())
    positive_xg_win = float(xg_leaders.result.eq("W").mean())
    # A predeclared descriptive relation, not a causal coefficient.
    xg_margin_relation = float(frame.xg_diff.corr(frame.goals - frame.conceded))
    top = league.sort_values("residual", ascending=False).head(3)
    xg = metrics["xg"]
    xg_rows = [{"model": name, **values["test"]} for name, values in xg["candidates"].items()]
    xg_rows.extend(
        [
            {"model": "constant_train_rate", **xg["baseline"]},
            {"model": "StatsBomb_external_reference", **xg["provider_reference"]},
        ]
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 160})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(league.xga, league.xg, c=league.points, cmap="viridis", s=80)
    for team, row in league.iterrows():
        axes[0].annotate(team, (row.xga, row.xg), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set(xlabel="xG conceded", ylabel="xG created", title="Premier League 2015/16: chance quality")
    sorted_residual = league.residual.sort_values()
    axes[1].barh(
        sorted_residual.index,
        sorted_residual,
        color=["#3b82f6" if v < 0 else "#65a30d" for v in sorted_residual],
    )
    axes[1].set(xlabel="Goals minus StatsBomb xG", title="Finishing residual (descriptive)")
    fig.tight_layout()
    fig.savefig(figures / "season.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    calibration = xg["calibration"]
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].plot(calibration["predicted"], calibration["observed"], "o-", color="#65a30d")
    axes[0].set(xlabel="Predicted probability", ylabel="Observed goal rate", title="Own xG: test calibration")
    axes[1].bar(
        [r["model"].replace("_", "\n") for r in xg_rows], [r["log_loss"] for r in xg_rows], color="#2563eb"
    )
    axes[1].set(ylabel="Log loss (lower is better)", title="Chronological held-out shots")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(figures / "models.png", bbox_inches="tight")
    plt.close(fig)
    report = f"""# Touchline Lab: факторы результатов АПЛ 2015/16

## Аннотация

Исследуем, какие наблюдаемые игровые показатели связаны с результатами матчей.
Выборка: {quality["matches"]} матчей, {quality["events"]:,} событий и {quality["shots"]:,} ударов.
Проект объединяет событийный анализ, собственную модель xG, интерпретируемую классификацию
исхода и LLM-отчёты с проверяемыми ссылками на события. Это наблюдательное исследование;
полученные связи не являются оценками причинного эффекта.

## Данные и контроль качества

Источник: [StatsBomb Open Data](https://github.com/hudl/open-data), коммит `{manifest["commit"]}`.
АПЛ 2015/16, competition_id=2, season_id=27. В исходном каталоге {manifest["available_matches"]} матчей.
SHA-256 каждого исходного JSON сохранён в `artifacts/manifest.json`. Исходные JSON сжаты
в локальном кэше, нормализованные события хранятся в SQLite. Сырые данные не входят в Git.

- Несоответствий событий официальному счёту: {quality["score_mismatches"]}.
- Ударов с недопустимым xG: {quality["invalid_shot_xg"]}.
- Ударов с координатами за пределами поля: {quality["invalid_shot_coordinates"]}.
  Их исходные значения сохранены, они исключены из обучения и оценки собственного xG,
  но включены в матчевую статистику и суммы провайдерского xG. В закреплённой версии
  событие `538f0ed8-576f-420c-adab-09e8c97c34b4` имеет x=120.2, y=1.0 (удар с углового).
  Грубых выходов координат более чем на единицу сетки: {quality["gross_coordinate_errors"]}.
- Автоголов: {quality["own_goals"]}; они учитываются через Own Goal Against и не дублируются.
- Дубли event_id запрещены первичным ключом, связи проверяются внешними ключами SQLite.
- Матчей у каждой команды: {int(frame.groupby("team").size().min())}–{int(frame.groupby("team").size().max())}.

## Вопросы и определения

1. Как связаны качество моментов и результат? xG — сумма вероятностей гола после ударов.
2. Как соотносятся фактические голы и xG за сезон? Разница не является чистой мерой удачи
   или мастерства: она также зависит от неопределённости и спецификации модели.
3. Какие признаки связаны с оценкой победы/ничьей/поражения? Используем разности xG, ударов,
   входов в штрафную, точности передач, событий давления, удалений и домашнее поле.

Вход в штрафную — точная передача или ведение из-за пределов зоны x ≥ 102, 18 ≤ y ≤ 62
внутрь неё на сетке StatsBomb 120 × 80. Pressure — зарегистрированное действие давления,
а не показатель успешности прессинга. Узлы сети передач — средние начала точных передач,
а не средние позиции игроков из трекинга.

## Описательные результаты

- Доля побед хозяев: {home_win:.1%} ({int((m.home_score > m.away_score).sum())} из {len(m)}).
- Команда с большим xG выиграла в {positive_xg_win:.1%} случаев ({int(xg_leaders.result.eq("W").sum())} из {len(xg_leaders)}).
- У {len(winners[winners.xg_diff < 0])} победителей xG был ниже, чем у соперника.
- Корреляция разницы xG с разницей голов: {xg_margin_relation:.3f}.
- Наибольшие положительные сезонные разницы голов и xG: {", ".join(f"{team} ({row.residual:+.2f})" for team, row in top.iterrows())}.

![Обзор сезона](figures/season.png)

Эти показатели описывают выборку. В корреляции использованы обе командные перспективы;
они зависимы, поэтому строки не трактуются как независимые наблюдения для проверки гипотез.
Результаты не доказывают, что дополнительные удары или давление сами по себе вызвали победу.

## Эксперимент 1: собственный xG

Цель — предсказать бинарный исход удара (гол / не гол). Доступные в момент удара признаки:
дистанция, угол, часть тела, техника, тип удара, давление и удар в одно касание.
Исход, конечные координаты удара, фактический счёт, имя игрока и StatsBomb xG исключены.
Пенальти включены как известный тип удара. Отдельная оценка пенальти и непенальти является
полезным следующим шагом; на малой выборке пенальти интервалы будут широкими.

Сравниваются Logistic Regression и HistGradientBoosting. Настройки моделей зафиксированы
в коде; выбор семейства выполнен по validation log loss. Простая база — частота голов на train.
StatsBomb xG показан как внешний ориентир: он использует более богатую информацию и
его обучающая выборка нам неизвестна, поэтому это не независимый конкурент с теми же данными.

Разделение по уникальным датам: train до {metrics["cutoffs"]["train_end"]}, validation до
{metrics["cutoffs"]["validation_end"]}, test — позже. Матчей:
{metrics["split_matches"]["train"]} / {metrics["split_matches"]["validation"]} / {metrics["split_matches"]["test"]}.
Ударов: {xg["train_shots"]} / {xg["validation_shots"]} / {xg["test_shots"]}.
Все события матча находятся в одной части. Преобразования обучены только на train.
Выбранная модель: **{xg["selected"]}**. Тест не используется для выбора.

{markdown_table(xg_rows, ["model", "log_loss", "brier", "roc_auc"])}

95% интервал test log loss выбранной модели: {xg["test_log_loss_ci95"][0]:.4f}–{xg["test_log_loss_ci95"][1]:.4f}.
Бутстрэп по матчам, 1000 повторов, seed=42. Калибровка построена по 8 квантильным группам.
Средний xG на test: {test_shots.model_xg.mean():.3f}; фактическая доля голов: {test_shots.goal.mean():.3f}.

![Оценка моделей](figures/models.png)

## Эксперимент 2: объяснение исхода сыгранного матча

Мультиномиальная логистическая регрессия: победа, ничья, поражение. Каждому матчу соответствуют
две командные строки. Разделение совпадает с первым экспериментом; обе строки остаются вместе.
Нормализация обучена на train, C выбирается из 0.1 / 1 / 10 по validation log loss.
Разница голов и сам результат не входят в признаки. Используется **провайдерский xG**;
собственный xG не подставляется в эту модель, чтобы не смешивать эксперименты.

{markdown_table([{"model": "logistic", **metrics["factors"]["test"]}, {"model": "train_class_frequencies", **metrics["factors"]["baseline"]}], ["model", "log_loss", "brier", "accuracy"])}

Для каждого матча вычисляются точные вклады признаков в логарифм отношения вероятностей
предпочтительного класса к следующему по вероятности. Сумма вкладов и intercept равна
log(P(класс 1) / P(класс 2)). Это объяснение модели, а не оценка причинного эффекта.
Модель использует итоговую статистику: её метрики **не являются качеством предматчевого прогноза**.

## Ход матча и условный эксперимент

Приложение показывает события при равном счёте, лидерстве и отставании. Гол засчитывается
после соответствующего события; автоголы учитываются один раз. Минуты считаются по времени
каждого тайма с добавленным временем. Показатели на 90 минут нестабильны на коротких отрезках.

Условный повтор тех же ударов рассчитывается точной свёрткой распределений Бернулли
(Poisson-binomial). Предполагаются независимость ударов и неизменный набор моментов;
не учитываются новые атаки, автоголы и изменения поведения после гола. Это не прогноз
реального переигранного матча и не вероятность «заслуженной» победы.

## Индивидуальные действия и линии команды

В основном разделе приложения добавлены отдельные показатели вратарей, защитников,
полузащитников и нападающих. В загруженной выборке есть {annotation_counts.get("Error", 0):,} событий Error,
{annotation_counts.get("Miscontrol", 0):,} Miscontrol и {annotation_counts.get("Dispossessed", 0):,} Dispossessed.
Это разные категории: только Error обозначается как явная ошибка
провайдера, а потеря мяча после отбора или рискованной передачи не равна доказанной ошибке.

Изучаются цепочки действий с первым ударом соперника в заданном окне (по умолчанию 20 с),
без перехода в другой тайм, возврата контроля и остановки игры. Один удар не суммируется
несколько раз в итогах по игрокам. Правило выбора последнего действия для таких итогов
является правилом отчётности, а не причинным распределением вины.

Реальный пример: Watford 1–2 Stoke City, 19 марта 2016, match_id=3753977.
У вратаря Heurelho da Silva Gomes зарегистрирована неточная передача в 50:37
(`417dbf67-0319-4f7f-ad1f-5c1ea58769b7`). Через 5.4 с в эпизоде соперника зарегистрирован
голевой удар xG=0.0679 (`494fed16-e21b-43cd-a4e0-f8ef2e0ecaf0`). Это полезный эпизод
для разбора неудачного распределения мяча, но он не доказывает, что только вратарь
стал причиной всего поражения. У самой передачи нет явного тега Error.

Положение невыбранных адресатов передач не наблюдается: отсутствие ключевого паса
не позволяет доказать упущенную доступную передачу. Для оценки ошибок вратаря на линии
недостаточно обычного xG; PSxG в этой функции не используется. Определения и границы
приведены в `docs/player-analysis.md`.

## LLM и проверка отчётов

В LLM передаются рассчитанные факты и UUID ключевых событий. Ответ имеет JSON-схему;
каждый раздел должен содержать существующие evidence_ids. Проверка отклоняет неизвестные
или пустые ссылки. В интерфейсе можно раскрыть факты и скачать отчёт. Без ключа API
показан явно обозначенный шаблон, а не имитация ответа LLM.

Наличие корректной ссылки само по себе не подтверждает логическую обоснованность вывода.
Рекомендуемый ручной протокол: выбрать матчи с победой фаворита, неожиданным результатом,
ничьей, удалением и автоголом; оценить точность чисел, поддержку каждого утверждения и
отсутствие выдуманной тактики. Живой API-вызов требует пользовательского ключа и модели.

## Ограничения и вывод

Один исторический сезон не гарантирует перенос на современный футбол. Межматчевая
зависимость команд полностью не устранена кластерным бутстрэпом. Обученные на ранних
матчах модели показывают для них in-sample оценки; только test используется для независимых
метрик. Freeze frames при ударах не являются непрерывным трекингом. Нет полной информации
о травмах, нагрузке, тренерских заданиях и позициях игроков между событиями.

Итог: можно воспроизводимо связать результат с наблюдаемыми моментами, продвижением мяча
и оценками моделей, а затем представить подтверждения пользователю. Причинные утверждения
потребуют отдельного дизайна исследования и более богатых данных.

## Воспроизведение

```bash
python -m touchline.download
python -m touchline.ingest
python -m touchline.modeling
python -m touchline.research
python -m pytest
streamlit run app.py
```

Data source: StatsBomb. Условия данных и официальный логотип: [исходный репозиторий](https://github.com/hudl/open-data).
"""
    (REPORTS / "research.md").write_text(report, encoding="utf-8")
    print(f"Research report generated; quality checks: {quality}")


if __name__ == "__main__":
    main()
