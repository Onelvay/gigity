# Manager review signals

The manager view separates **contribution** from **review priority**. A player can have a
green strong metric and a red documented episode at the same time. Neither colour is a
whole-player verdict or an instruction to drop, sell or replace a player.

## Green: observed strength

For each role/profile, compare the player's season per-90 value to EPL 2015/16 players
with at least 450 minutes in the same profile. Percentages use their original denominators.
Use midpoint ranks for ties. A positive metric with percentile ≥0.75 is a green signal.
Zero/missing metrics do not become green, and players below 450 minutes receive no such
comparative strength signal. The actual value, peer median and seasonal total are shown.

The comparison is descriptive. Volume can depend on team style and workload; goalkeeper
save percentage is unadjusted for shot difficulty. Green is evidence for a specific metric,
not proof of universally superior player quality. The peer profiles and positive-feature
lists are explicit in `touchline/manager.py`.

## Yellow: attention

- Pass completion below the role's 25th percentile, with at least 100 attempted passes.
- Poor-control/dispossession events on the player's own half: at least five and a per-90
  value in the role's upper quartile.
- Fouls near own penalty area: at least three and an upper-quartile per-90 rate.
- At least five yellow-card annotations.
- Non-penalty finishing residual ≤−2 with at least 20 shots.
- One missed penalty, explicit failed goalkeeper handling, or provider Error without a
  linked scoring shot.
- A single distinct opposition scoring shot after a selected action.
- Goalkeeper goals allowed from pre-shot xG≤0.10: review only; no proof of an easy save.

## Red: prioritise episode review

- Penalty conceded, dismissal or own goal.
- At least two missed penalties; display scored/taken counts and note the small sample.
- Provider Error annotation followed by a linked scoring shot.
- At least two distinct linked opposition scoring shots after selected actions.

These thresholds are review rules, not learned causal costs. An action before a scoring
shot does not establish individual responsibility for the goal or match defeat. Linked
shots are deduplicated by shot ID per player. Low pre-shot xG alone never turns a keeper
red. A small-sample player without a specific negative event remains grey, not green.

## Action and evidence

Each signal has a short review/training task. Event-based signals retain original event
and match IDs; the UI opens dates, times, opponents and raw source links. Poor-control
signals cite only the own-half actions matching the metric. Green season comparisons
are reproducible from `player_matches` and `leicester_players` in SQLite.

Generate review CSVs with `python -m touchline.manager`. This produces the summary and
all signals without requiring an LLM or API access. AI explanation remains a separate,
optional interpretation of evidence.
