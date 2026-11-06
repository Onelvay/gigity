PRAGMA foreign_keys = ON;
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY, date TEXT NOT NULL, kick_off TEXT NOT NULL,
    home_id INTEGER NOT NULL, home TEXT NOT NULL, away_id INTEGER NOT NULL, away TEXT NOT NULL,
    home_score INTEGER NOT NULL, away_score INTEGER NOT NULL, match_week INTEGER,
    stadium TEXT, source_commit TEXT NOT NULL
);
CREATE TABLE events (
    event_id TEXT PRIMARY KEY, match_id INTEGER NOT NULL REFERENCES matches(match_id),
    event_index INTEGER NOT NULL, period INTEGER NOT NULL, minute INTEGER NOT NULL,
    second INTEGER NOT NULL, clock REAL NOT NULL, type TEXT NOT NULL,
    team_id INTEGER, team TEXT, player_id INTEGER, player TEXT,
    possession INTEGER, possession_team_id INTEGER, x REAL, y REAL, end_x REAL, end_y REAL,
    outcome TEXT, complete INTEGER, recipient_id INTEGER, recipient TEXT,
    under_pressure INTEGER, card TEXT, play_pattern TEXT
);
CREATE INDEX idx_events_match ON events(match_id, event_index);
CREATE INDEX idx_events_team ON events(team_id, type);
CREATE TABLE shots (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id), match_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL, player TEXT, minute INTEGER, period INTEGER,
    clock REAL NOT NULL, x REAL NOT NULL, y REAL NOT NULL,
    distance REAL NOT NULL, angle REAL NOT NULL, body_part TEXT NOT NULL,
    technique TEXT NOT NULL, shot_type TEXT NOT NULL, under_pressure INTEGER NOT NULL,
    first_time INTEGER NOT NULL, goal INTEGER NOT NULL, statsbomb_xg REAL NOT NULL
);
CREATE INDEX idx_shots_match ON shots(match_id);

CREATE VIEW team_totals AS
SELECT match_id, team_id, MAX(team) AS team,
       SUM(type = 'Shot') AS shots,
       SUM(type = 'Pressure') AS pressures,
       SUM(type = 'Pass') AS passes,
       SUM(type = 'Pass' AND complete = 1) AS completed_passes,
       SUM(type IN ('Pass', 'Carry') AND complete = 1
           AND end_x >= 102 AND end_y BETWEEN 18 AND 62
           AND NOT (x >= 102 AND y BETWEEN 18 AND 62)) AS box_entries,
       COALESCE(SUM(card IN ('Red Card', 'Second Yellow')), 0) AS red_cards
FROM events WHERE period <= 2 GROUP BY match_id, team_id;

CREATE VIEW team_match AS
WITH sides AS (
    SELECT match_id, home_id AS team_id, home AS team, away_id AS opponent_id,
           away AS opponent, home_score AS goals, away_score AS conceded, 1 AS is_home FROM matches
    UNION ALL
    SELECT match_id, away_id, away, home_id, home, away_score, home_score, 0 FROM matches
), xg AS (
    SELECT match_id, team_id, SUM(statsbomb_xg) AS xg
    FROM shots WHERE period <= 2 GROUP BY match_id, team_id
)
SELECT m.date, s.*, t.shots, t.passes, t.completed_passes, t.pressures, t.box_entries, t.red_cards,
       COALESCE(xg.xg, 0) AS xg,
       CASE WHEN s.goals > s.conceded THEN 'W' WHEN s.goals = s.conceded THEN 'D' ELSE 'L' END AS result,
       CASE WHEN s.goals > s.conceded THEN 3 WHEN s.goals = s.conceded THEN 1 ELSE 0 END AS points
FROM sides s JOIN matches m USING(match_id)
LEFT JOIN team_totals t ON t.match_id = s.match_id AND t.team_id = s.team_id
LEFT JOIN xg ON xg.match_id = s.match_id AND xg.team_id = s.team_id;
