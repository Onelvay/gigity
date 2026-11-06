# Model card

## Intended use

Educational and portfolio research on Premier League 2015/16. The application describes
completed matches and supports inspection of individual events. It is not evaluated as a
betting system, scouting system or predictor of future seasons.

## Models

**Own xG:** binary shot-goal classifier. Logistic Regression and HistGradientBoosting are
compared with fixed settings and chosen by chronological validation log loss. Numerical
inputs: distance to goal centre, goal opening angle, pressure and first-time flag.
Categorical inputs: body part, technique, shot type. Penalties are included. Missing optional
flags follow StatsBomb's event encoding (absent means false); missing categories are Unknown.
Shots outside the 120×80 grid are excluded from own-xG training/evaluation and left unscored
by this model in the application. Source values remain in the database.

**Post-match outcome:** StandardScaler + multinomial LogisticRegression; C is selected from
0.1, 1 and 10 using validation log loss. Inputs: team-minus-opponent provider xG, shots,
penalty-area entries, pass-completion rate, recorded Pressure events, red cards and home status.
Target: W/D/L from the selected team's perspective. Final goals are used to construct targets
and are not predictors. Two rows per match remain grouped in evaluation.

## Evaluation

Chronological 60/20/20 split of unique match dates, exact cutoffs in `artifacts/metrics.json`.
All preprocessing fits only train. Models are not refitted on train+validation before published
test evaluation. Test metrics are reported once by the reproducible experiment command;
rerunning produces the same protocol, not a new independent experiment.

Bootstrap confidence intervals resample whole test matches (1000 replicates, seed 42), preserving
within-match dependence but not all between-match team dependence. Accuracy for rare binary
goals can be misleading; log loss, Brier score, ROC AUC and calibration are also shown.

Binary Brier is mean((p-goal)^2). Multiclass Brier is the mean sum of squared class errors.
These two Brier values have different scales and are not compared across tasks.

## Explanation contract

For a row, identify the highest and second-highest predicted classes. For standardized feature
z_j, show z_j*(coefficient_top_j - coefficient_reference_j). The intercept difference plus all
contributions reconstructs log(P(top)/P(reference)) exactly. Values are relative to the training
feature means and the named class contrast; they are not percentage-point effects or causal estimates.
Correlated predictors can redistribute apparent importance.

## Known limitations

- Only one historical season; evaluation is within-season, not external validation.
- Available shot features are much less detailed than the provider model's context.
- The training set's displayed match estimates are in-sample; the UI identifies the split.
- Observational end-of-match features are affected by score state, opponent strength and tactics.
- Own goals contribute to official score but not shot xG. Outcome classification cannot perfectly
  explain these through chance-quality inputs alone.
- Shot independence in fixed-shot replay is a simplifying assumption, especially for rebounds.
- No continuous tracking data; shot freeze frames do not fill this gap.
