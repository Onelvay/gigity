# Per-metric colours

Both the main player-stat table and the expanded full-stat table give every row a colour,
text status and explanation. Header cards are coloured too; their explanation is in the help tooltip.

| Colour | Meaning |
| --- | --- |
| Green | Upper-quartile beneficial metric within the same EPL role/profile, a positive recorded outcome, or no recorded adverse event |
| Yellow | Lower-quartile beneficial metric, high frequency of adverse events, or an episode requiring review |
| Red | A critical documented event or an existing evidence-based review-priority signal |
| Blue | Middle-range comparison, workload/context without a good/bad direction, or no relevant opportunity |
| Grey | Missing denominator, fewer than 450 minutes for a comparison, unavailable peer data, or a metric outside the player's role |

Counting comparisons use per-90 values; percentages and average TL use their original scales.
Peer comparisons use the same season rating profile and 450+ minutes. Midpoint ties do not
create an artificial green signal when all peers have zero. For adverse frequency counters,
the upper quartile is yellow, not green. No penalty attempts is blue, not a perfect success rate.

Goals conceded and shots faced are context counts. Goals allowed from low ordinary xG are
yellow review flags, never automatic red keeper errors. Outfield players' keeper-only numbers
are grey. A red provider Error signal remains red when its metric is shown in the full table.

Green for zero incidents describes the absence of that event, not overall player quality.
The UI legend and the expandable colour-reason table make this distinction explicit.
