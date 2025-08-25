# IPL Player Performance & Season-wise Analysis

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# Download IPL dataset
dataset_dir = kagglehub.dataset_download("patrickb1912/ipl-complete-dataset-20082020")
print("Dataset directory:", dataset_dir)
print("Files in dataset:", os.listdir(dataset_dir))

# Load matches and deliveries CSV
matches = pd.read_csv(os.path.join(dataset_dir, "matches.csv"))
deliveries = pd.read_csv(os.path.join(dataset_dir, "deliveries.csv"))

# Strip extra spaces from column names
matches.columns = matches.columns.str.strip()
deliveries.columns = deliveries.columns.str.strip()

# Overall Batting Analysis
batting = deliveries.groupby("batter").agg(
    runs_scored=("batsman_runs", "sum"),
    balls_faced=("batsman_runs", "count"),
    innings_played=("match_id", "nunique")
).reset_index()

batting["strike_rate"] = (batting["runs_scored"] / batting["balls_faced"]) * 100

dismissals = deliveries[deliveries["player_dismissed"].notnull()]
outs = dismissals.groupby("player_dismissed").size().reset_index(name="outs")
batting = batting.merge(outs, left_on="batter", right_on="player_dismissed", how="left")
batting["outs"] = batting["outs"].fillna(0)
batting["average"] = batting.apply(
    lambda row: row["runs_scored"]/row["outs"] if row["outs"]>0 else row["runs_scored"], axis=1
)

batting["consistency_score"] = batting["average"]*0.6 + batting["strike_rate"]*0.4
top_batsmen = batting.sort_values("consistency_score", ascending=False).head(10)
print("\nTop 10 Consistent Batsmen Overall:\n", top_batsmen[["batter","runs_scored","average","strike_rate","consistency_score"]])

# Overall Bowling Analysis
bowling = deliveries.groupby("bowler").agg(
    runs_conceded=("total_runs", "sum"),
    balls_bowled=("ball", "count"),
    wickets=("player_dismissed", lambda x: x.notnull().sum())
).reset_index()

bowling["overs"] = bowling["balls_bowled"]//6 + (bowling["balls_bowled"]%6)/6
bowling["economy"] = bowling["runs_conceded"]/bowling["overs"]
bowling["bowling_strike_rate"] = bowling.apply(
    lambda row: row["balls_bowled"]/row["wickets"] if row["wickets"]>0 else None, axis=1
)

top_bowlers = bowling.sort_values("economy").head(10)
print("\nTop 10 Bowlers Overall by Economy Rate:\n", top_bowlers[["bowler","runs_conceded","balls_bowled","wickets","economy"]])

# Season-wise Analysis
# Merge deliveries with season info from matches
deliveries = deliveries.merge(matches[['id','season']], left_on='match_id', right_on='id', how='left')

# Season-wise Batting
season_batting = deliveries.groupby(['season','batter']).agg(
    runs_scored=('batsman_runs','sum'),
    balls_faced=('batsman_runs','count'),
    innings_played=('match_id','nunique')
).reset_index()

season_batting['strike_rate'] = (season_batting['runs_scored']/season_batting['balls_faced'])*100

dismissals = deliveries[deliveries['player_dismissed'].notnull()]
outs = dismissals.groupby(['season','player_dismissed']).size().reset_index(name='outs')
season_batting = season_batting.merge(outs, left_on=['season','batter'], right_on=['season','player_dismissed'], how='left')
season_batting['outs'] = season_batting['outs'].fillna(0)
season_batting['average'] = season_batting.apply(
    lambda row: row['runs_scored']/row['outs'] if row['outs']>0 else row['runs_scored'], axis=1
)
season_batting['consistency_score'] = season_batting['average']*0.6 + season_batting['strike_rate']*0.4

top_batsmen_season = season_batting.sort_values(['season','consistency_score'], ascending=[True,False]).groupby('season').head(3)
print("\nTop 3 Batsmen per Season:\n", top_batsmen_season[['season','batter','runs_scored','average','strike_rate','consistency_score']])

# Season-wise Bowling
season_bowling = deliveries.groupby(['season','bowler']).agg(
    runs_conceded=('total_runs','sum'),
    balls_bowled=('ball','count'),
    wickets=('player_dismissed', lambda x: x.notnull().sum())
).reset_index()

season_bowling['overs'] = season_bowling['balls_bowled']//6 + (season_bowling['balls_bowled']%6)/6
season_bowling['economy'] = season_bowling['runs_conceded']/season_bowling['overs']
season_bowling['bowling_strike_rate'] = season_bowling.apply(
    lambda row: row['balls_bowled']/row['wickets'] if row['wickets']>0 else None, axis=1
)

top_bowlers_season = season_bowling.sort_values(['season','economy'], ascending=[True,True]).groupby('season').head(3)
print("\nTop 3 Bowlers per Season (Economy Rate):\n", top_bowlers_season[['season','bowler','runs_conceded','wickets','economy']])

# Visualization

# Top 10 Batsmen Overall
plt.figure(figsize=(10,6))
plt.bar(top_batsmen["batter"], top_batsmen["runs_scored"], color="skyblue")
plt.title("Top 10 Consistent Batsmen Overall")
plt.ylabel("Runs Scored")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Top 10 Bowlers Overall
plt.figure(figsize=(10,6))
plt.bar(top_bowlers["bowler"], top_bowlers["economy"], color="orange")
plt.title("Top 10 Bowlers Overall by Economy Rate")
plt.ylabel("Economy Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Season-wise Top Batsmen Heatmap
pivot_batting = top_batsmen_season.pivot(index='batter', columns='season', values='runs_scored').fillna(0)
plt.figure(figsize=(12,6))
sns.heatmap(pivot_batting, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Top 3 Batsmen Runs per Season")
plt.ylabel("Batter")
plt.xlabel("Season")
plt.show()

# Season-wise Top Bowlers Heatmap
pivot_bowling = top_bowlers_season.pivot(index='bowler', columns='season', values='economy').fillna(0)
plt.figure(figsize=(12,6))
sns.heatmap(pivot_bowling, annot=True, fmt=".2f", cmap="OrRd_r")
plt.title("Top 3 Bowlers Economy per Season")
plt.ylabel("Bowler")
plt.xlabel("Season")
plt.show()
