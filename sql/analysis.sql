-- These queries power the season view; no hard-coded results.
-- League table with expected-goal differential and finishing residual.
SELECT team, COUNT(*) AS played, SUM(points) AS points,
       SUM(result = 'W') AS wins, SUM(result = 'D') AS draws, SUM(result = 'L') AS losses,
       SUM(goals) AS goals, SUM(conceded) AS conceded,
       ROUND(SUM(xg), 2) AS xg,
       ROUND(SUM(goals - xg), 2) AS finishing_residual
FROM team_match GROUP BY team ORDER BY points DESC, SUM(goals-conceded) DESC, goals DESC;

-- Prior five games only: the current game's result is excluded by the frame.
SELECT date, team, opponent, points,
       AVG(points) OVER (PARTITION BY team_id ORDER BY date, match_id
                        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS prior_five_ppg
FROM team_match ORDER BY team, date;
