"""Chronological experiments. No score, shot outcome or provider xG in own-xG inputs."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from touchline.config import ARTIFACTS, directories
from touchline.data import factor_frame, matches, query

SHOT_NUM = ["distance", "angle", "under_pressure", "first_time"]
SHOT_CAT = ["body_part", "technique", "shot_type"]
SHOT_FEATURES = SHOT_NUM + SHOT_CAT
FACTORS = [
    "xg_diff",
    "shots_diff",
    "box_entries_diff",
    "completion_diff",
    "pressures_diff",
    "red_cards_diff",
    "is_home",
]
LABELS = {
    "xg_diff": "Разница xG",
    "shots_diff": "Разница ударов",
    "box_entries_diff": "Входы в штрафную",
    "completion_diff": "Точность передач",
    "pressures_diff": "Действия давления",
    "red_cards_diff": "Разница удалений",
    "is_home": "Домашнее поле",
}


def split_dates(match_frame):
    dates = sorted(match_frame.date.unique())
    if len(match_frame) < 100 or len(dates) < 30:
        raise ValueError(
            "Для оценки моделей нужно минимум 100 матчей и 30 игровых дат. Загрузите весь сезон."
        )
    train_end, val_end = dates[int(len(dates) * 0.6) - 1], dates[int(len(dates) * 0.8) - 1]
    return train_end, val_end


def masks(frame, cutoffs):
    train_end, val_end = cutoffs
    return frame.date <= train_end, (frame.date > train_end) & (frame.date <= val_end), frame.date > val_end


def scores(y, probability, classes):
    y = np.asarray(y)
    probability = np.asarray(probability)
    target = (y[:, None] == np.asarray(classes)[None, :]).astype(float)
    result = {
        "log_loss": float(log_loss(y, probability, labels=classes)),
        "brier": float(np.mean(np.sum((target - probability) ** 2, axis=1))),
        "accuracy": float(accuracy_score(y, np.asarray(classes)[probability.argmax(axis=1)])),
    }
    # For binary tasks report the conventional one-column Brier score.
    if len(classes) == 2:
        result["brier"] /= 2
        result["roc_auc"] = float(roc_auc_score(y, probability[:, 1]))
    return result


def loss_interval(frame, y, probability, classes, repetitions=1000):
    """Match-cluster bootstrap: keep shots / both team-perspectives together."""
    lookup = {value: i for i, value in enumerate(classes)}
    indices = np.array([lookup[value] for value in y])
    losses = -np.log(np.clip(probability[np.arange(len(y)), indices], 1e-12, 1))
    grouped = pd.DataFrame({"match_id": frame.match_id.to_numpy(), "loss": losses}).groupby("match_id")
    sums, counts = grouped.loss.sum().to_numpy(), grouped.loss.count().to_numpy()
    samples = np.random.default_rng(42).integers(0, len(sums), size=(repetitions, len(sums)))
    means = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def shot_pipeline(classifier):
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    pre = ColumnTransformer(
        [
            ("numeric", numeric, SHOT_NUM),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SHOT_CAT),
        ]
    )
    return make_pipeline(pre, classifier)


def train_xg(shot_frame, cutoffs):
    train, val, test = masks(shot_frame, cutoffs)
    x, y = shot_frame[SHOT_FEATURES], shot_frame.goal
    candidates = {
        "logistic": shot_pipeline(LogisticRegression(C=1, max_iter=2000, random_state=42)),
        "gradient_boosting": shot_pipeline(
            HistGradientBoostingClassifier(
                max_iter=150,
                max_leaf_nodes=7,
                l2_regularization=5,
                learning_rate=0.05,
                early_stopping=False,
                random_state=42,
            )
        ),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(x[train], y[train])
        results[name] = {
            "validation": scores(y[val], model.predict_proba(x[val]), model.classes_),
            "test": scores(y[test], model.predict_proba(x[test]), model.classes_),
        }
    selected = min(results, key=lambda key: results[key]["validation"]["log_loss"])
    model = candidates[selected]
    base_rate = float(y[train].mean())
    baseline = np.tile([1 - base_rate, base_rate], (test.sum(), 1))
    provider = shot_frame.loc[test, "statsbomb_xg"].to_numpy().clip(1e-8, 1 - 1e-8)
    probability = model.predict_proba(x[test])
    actual, predicted = calibration_curve(y[test], probability[:, 1], n_bins=8, strategy="quantile")
    metrics = {
        "selected": selected,
        "candidates": results,
        "baseline": scores(y[test], baseline, [0, 1]),
        "provider_reference": scores(y[test], np.column_stack([1 - provider, provider]), [0, 1]),
        "test_log_loss_ci95": loss_interval(shot_frame[test], y[test], probability, [0, 1]),
        "calibration": {"predicted": predicted.tolist(), "observed": actual.tolist()},
        "train_shots": int(train.sum()),
        "validation_shots": int(val.sum()),
        "test_shots": int(test.sum()),
        "features": SHOT_FEATURES,
    }
    joblib.dump(model, ARTIFACTS / "xg.joblib")
    predictions = shot_frame[["event_id", "match_id", "date", "goal", "statsbomb_xg"]].copy()
    predictions["model_xg"] = model.predict_proba(x)[:, 1]
    predictions["split"] = np.select([train, val], ["train", "validation"], default="test")
    predictions.to_csv(ARTIFACTS / "shot_predictions.csv", index=False)
    return metrics


def train_factors(frame, cutoffs):
    train, val, test = masks(frame, cutoffs)
    x, y = frame[FACTORS], frame.result
    candidates, results = {}, {}
    for c in (0.1, 1, 10):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=42))
        model.fit(x[train], y[train])
        key = str(c)
        candidates[key] = model
        results[key] = scores(y[val], model.predict_proba(x[val]), model.classes_)
    selected = min(results, key=lambda key: results[key]["log_loss"])
    model = candidates[selected]
    classes = model.classes_
    priors = y[train].value_counts(normalize=True).reindex(classes).to_numpy()
    probability = model.predict_proba(x[test])
    metrics = {
        "selected_C": float(selected),
        "validation_candidates": results,
        "test": scores(y[test], probability, classes),
        "baseline": scores(y[test], np.tile(priors, (test.sum(), 1)), classes),
        "test_log_loss_ci95": loss_interval(frame[test], y[test], probability, classes),
        "classes": classes.tolist(),
        "features": FACTORS,
        "test_matches": int(frame[test].match_id.nunique()),
        "coefficients": dict(zip(classes, model[-1].coef_.tolist(), strict=False)),
    }
    joblib.dump(model, ARTIFACTS / "factors.joblib")
    return metrics


def explain_factors(model, row):
    """Exact additive log-odds contrast: predicted class vs runner-up."""
    x = pd.DataFrame([row])[FACTORS]
    probability = model.predict_proba(x)[0]
    best, reference = np.argsort(probability)[-2:][::-1]
    z = model[0].transform(x)[0]
    contribution = z * (model[-1].coef_[best] - model[-1].coef_[reference])
    intercept = float(model[-1].intercept_[best] - model[-1].intercept_[reference])
    return {
        "predicted": str(model.classes_[best]),
        "reference": str(model.classes_[reference]),
        "probabilities": dict(zip(model.classes_, probability.tolist(), strict=False)),
        "intercept": intercept,
        "contributions": dict(zip(FACTORS, contribution.tolist(), strict=False)),
    }


def main():
    directories()
    match_frame = matches()
    cutoffs = split_dates(match_frame)
    shot_frame = query(
        "SELECT s.*, m.date FROM shots s JOIN matches m USING(match_id) WHERE s.period <= 2 "
        "AND s.x BETWEEN 0 AND 120 AND s.y BETWEEN 0 AND 80"
    )
    metrics = {
        "cutoffs": {"train_end": cutoffs[0], "validation_end": cutoffs[1]},
        "match_count": len(match_frame),
        "shot_count": len(shot_frame),
    }
    metrics["xg"] = train_xg(shot_frame, cutoffs)
    print("Own xG experiment complete", flush=True)
    metrics["factors"] = train_factors(factor_frame(), cutoffs)
    metrics["split_matches"] = {
        name: int(mask.sum())
        for name, mask in zip(("train", "validation", "test"), masks(match_frame, cutoffs), strict=False)
    }
    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
