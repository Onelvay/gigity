# Leicester season ratings (TL)

TL is an engineered descriptive index, not an official/expert rating, a trained prediction,
or a measure of causal responsibility. The formula and coefficients are explicit so that
readers can challenge the choices. A narrow difference such as 6.03 vs 6.10 should not be
treated as a reliable difference in player quality.

## Peer group and base score

Each match appearance of 30+ regulation minutes is compared with appearances in the same
profile by EPL 2015/16 players who accumulated 450+ season minutes. Counting metrics are
converted to per-90 rates. Pass-completion and save rates use their actual denominators.
Within-feature percentile uses the midpoint of ties. Missing percentage denominators receive
the neutral 0.5 percentile rather than an invented perfect or zero rate.

The four position groups remain visible in the UI. Attacking/left/right midfield positions
use a separate attacking-midfield rating profile, so Mahrez is not evaluated primarily on
defensive recoveries. Profile weights are in `touchline/season.py` and the app's method view.
They sum to one. A player's season role/profile is the one with the most played minutes.

```
base = 6 + 6 × (weighted percentile − 0.5)
bonus = min(1, 0.6 × penalties saved + 0.2 × penalties won)
match_rating = clip(base + bonus − episode deductions, 0, 10)
season_rating = sum(rating × minutes) / sum(minutes in rated appearances)
```

Appearances under 30 minutes are retained in season counts but not assigned a rating.
Players under 450 minutes are labelled a small sample and excluded from the default ranking.
Regulation minutes are calculated from lineup, substitution and dismissal boundaries,
clamped to 45/90 minutes. Stoppage time is not added. Temporary exits are not subtracted.

## Deductions

| Action | Deduction |
| --- | ---: |
| Penalty conceded | 0.80 |
| Red/second-yellow dismissal | 0.80 |
| Own goal | 0.70 |
| Candidate action followed by opposition scoring shot | 0.50 |
| Missed penalty, including a saved penalty | 0.40 |
| Provider Error or explicitly failed goalkeeper handling/sweep | 0.35 |
| Yellow card | 0.10 |
| Foul near own penalty area | 0.06 |
| Miscontrol | 0.04 |
| Dribbled past | 0.035 |
| Dispossessed | 0.025 |
| Failed take-on | 0.02 |
| Other foul | 0.015 |
| Incomplete pass | 0.005 |
| Candidate action before a non-goal opposition shot | 0.15 × shot xG |

For each player and linked shot/event episode, only the maximum applicable deduction is
charged. Penalty + red card + resulting goal is therefore 0.80, not 2.10. Every charged
episode ID, value and reason is available under the player's match-rating details.
These are chosen presentation weights, not empirically estimated costs or lost points.

## Goalkeepers

Saves are identified from opposition shots with Saved / Saved to Post outcomes while the
keeper was on the field. Save percentage divides these by goals + those saved shots; own
goals are not classified as opponent shots. Goals conceded include own goals if applicable.
Penalty saves/allowed and explicitly failed collection/punch/smother/sweep events are separate.

Goals conceded from pre-shot xG ≤0.10 are review flags and carry no additional deduction.
There is no PSxG in this feature; neither the flag nor save percentage demonstrates that a
specific shot was easy or that the goalkeeper was at fault. A keeper's TL index is especially
limited by shot-difficulty information.

## Foul consequences

After a Foul Committed annotation, the paired Foul Won within two seconds is skipped.
A penalty links to the next opposition penalty shot in the same period before other
controlled play. Other fouls link to the first shot in the free-kick/advantage possession.
The search stops at a new possession, restart, foul, regained control, half end or 180 seconds.
It does not claim that all later goals were caused by the foul or evaluate unobserved marking.

Unsuccessful actions can be reasonable risks. These penalties deliberately make the rating
sensitive to documented negative events, but remain contestable. Team style, opponent quality,
off-ball positioning and changes of role are not fully controlled. No alternative passing
decision is labelled a missed key pass without tracking/video evidence.
