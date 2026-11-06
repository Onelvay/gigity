# LLM report evaluation

The LLM explains a precomputed evidence packet. It does not calculate match metrics or search
for injuries, tactics or external context. The API integration uses structured output and validates
that every section cites at least one existing evidence ID. This is a source-link check, not a
semantic entailment check.

## Manual evaluation protocol

After configuring your model, select at least 10 held-out matches spanning wins, draws,
upsets, large xG/result discrepancies, red cards and own goals. Record model ID, date,
prompt version (Git commit), match ID and output. Independently score each report:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Numeric accuracy | Incorrect or invented numbers | Rounding/wording ambiguity | All figures match evidence |
| Evidence support | Unsupported central claims | Some overinterpretation | Every factual claim supported |
| Scope discipline | Invented injuries/tactics | Ambiguous claims | Observations and hypotheses separated |
| Usefulness | Generic restatement | Some relevant synthesis | Clear synthesis of the match's key contrasts |
| Limitations | Missing or misleading | Generic caution | Relevant uncertainty and scope stated |

Record all outputs, including failures. Report average and minimum score per dimension,
reference-validation failure rate, latency and number of attempts. For a stronger study, use
two raters and discuss disagreements. Do not invent completed evaluation scores: this manual
evaluation has not been executed without a configured API model.

## Automated checks

The test suite verifies that fabricated and empty evidence references are rejected. A template
is provided without claiming it came from an LLM. Source event UUIDs are preserved in exports.
An LLM may cite a valid ID while drawing an unsupported conclusion, so human review remains necessary.
