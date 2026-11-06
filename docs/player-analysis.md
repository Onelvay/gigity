# Player and position-group review

The primary match tab now reviews individual players and four position groups. It uses
the original event annotations, the Starting XI, substitutions and red cards. No universal
0–10 performance grade is fabricated from event totals.

## Role-specific facts

- **Goalkeepers:** recorded Shot Saved / Shot Saved to Post events, goals conceded while
  on the field, passing accuracy and unsuccessful distribution. Conceded goals include own
  goals; save-event counts are not an adjusted save percentage. Ordinary pre-shot xG cannot
  establish goalkeeper shot-stopping errors. Post-shot xG is not supplied by this feature.
- **Defenders:** tackle attempts and successful outcomes, interception events, blocks,
  clearances, dribbled-past events and subsequent attacking sequences. An interception event
  is not necessarily retained possession; a block may block a pass as well as a shot.
- **Midfielders:** pass completion, completed passes gaining at least 10 x-coordinate units,
  passes leading to shots, xA, poor touches and dispossessions. The progression definition
  is an explicit simple x-gain rule, not a provider's proprietary progressive-pass metric.
- **Forwards:** goals, shots, provider xG, unconverted chances, key passes, xA and assists.
  Missed-shot xG does not prove poor finishing on an individual attempt.

Roles use the most frequent annotated position in a player's actions, falling back to the
lineup/substitution position. Wing backs are defenders; wingers are forwards; attacking
midfielders remain midfielders. A player changing positions is represented by one dominant
position for this match, rather than exact minutes at every position. Minutes include
stoppage time, use lineup/substitution/dismissal boundaries and do not subtract temporary exits.

A key pass is an actual annotated shot assist or a pass linked to a shot ID. xA sums provider
xG of the linked shots. These measure executed passes, not available alternatives.

## What counts as an incident

The interface keeps the following categories separate:

| Source annotation | Display |
| --- | --- |
| Error | Explicit provider-annotated error |
| Miscontrol | Poor control / bad touch |
| Dispossessed | Opponent tackled the ball away |
| Dribbled Past | Opponent dribbled past the player |
| Dribble / Incomplete | Unsuccessful take-on |
| Pass / Incomplete | Incomplete pass |
| Pass / Out | Pass out of play |
| Pass / Pass Offside | Pass associated with offside |

Only the first category is labelled an explicit provider Error. Other unsuccessful actions
may be reasonable risks, good defending or context-dependent outcomes, not individual mistakes.
Unknown pass outcomes are excluded from the unsuccessful-pass category.

## Linking an action to an opposition shot

The configurable window is 10, 15, 20 or 30 seconds. In event order, look for the first
opposition shot within the same period and one opposition possession, stopping at:

- the reviewed team's next controlled Pass, Carry, Dribble or Shot in its own possession;
- a foul, offside, half end, ball-out event, or a recognised restart pass;
- a new opposition possession after its first controlled action;
- expiry of the time window.

An initial out-of-play or offside pass is not linked through the ensuing restart. A Dribbled
Past or Error event may occur during an already ongoing opposition possession. The feature
therefore says an attack followed the action, not necessarily that a new turnover happened.

Several reviewed actions can precede one shot. All remain inspectable, but player/line totals
count that shot once per reviewed team, at the most recent preceding candidate action. This
deduplication is a reporting rule, **not a defensible causal attribution of responsibility**.
The feature considers the first shot, not total xG of every possible rebound sequence.

The replay map transforms opposition coordinates into the reviewed team's reference direction.
It shows event locations, not continuous ball paths or off-ball positions. There is no video feed.

## What cannot be concluded

No key pass does not mean a player ignored a free teammate. This requires the positions of
all players, passing lanes and pressure at decision time, usually supported by video. This
Premier League season does not provide continuous tracking or 360 coverage for this feature.
Likewise, a conceded goal alone does not establish a goalkeeper error, an unrecorded marking
mistake or the sole player responsible for a defeat.

## Sources

Definitions of poor control and dispossession are consistent with the provider's
[event descriptions](https://live-data-api-guide.statsbomb.com/api-reference/event-descriptions.html).
The distinction between pre-shot xG and goalkeeper evaluation is discussed in StatsBomb's
[post-shot xG article](https://blogarchive.statsbomb.com/articles/soccer/a-new-way-to-measure-keepers-shot-stopping-post-shot-expected-goals/).
Actual observations and UUIDs come from the pinned local StatsBomb Open Data files.
