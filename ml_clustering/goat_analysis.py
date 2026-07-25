import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# ---- Load Data ----
df = pd.read_csv('data.csv')

# ---- Career Stats Comparison ----
goals_per_player = df.groupby('Player').size().reset_index(name='Total_Goals')
seasons_per_player = df.groupby('Player')['Season'].nunique().reset_index(name='Total_Seasons')
stats = goals_per_player.merge(seasons_per_player, on='Player')
stats['Goals_Per_Season'] = (stats['Total_Goals'] / stats['Total_Seasons']).round(1)
print(stats)

# ---- Feature Engineering per Season ----
season_stats = df.groupby(['Player', 'Season']).agg(
    Goals=('Player', 'count'),
    Penalties=('Type', lambda x: (x == 'Penalty').sum()),
    Headers=('Type', lambda x: (x == 'Header').sum()),
    FreeKicks=('Type', lambda x: (x == 'Direct free kick').sum()),
    LeftFoot=('Type', lambda x: (x == 'Left-footed shot').sum()),
    RightFoot=('Type', lambda x: (x == 'Right-footed shot').sum()),
    AwayGoals=('Venue', lambda x: (x == 'A').sum()),
).reset_index()

# ---- Normalize + Cluster ----
features = ['Goals', 'Penalties', 'Headers', 'FreeKicks', 'LeftFoot', 'RightFoot', 'AwayGoals']
scaler = StandardScaler()
X = scaler.fit_transform(season_stats[features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
season_stats['Cluster'] = kmeans.fit_predict(X)

# ---- Cluster Profile Summary (important for resume + app) ----
cluster_summary = season_stats.groupby('Cluster')[features].mean().round(2)
print("\n=== Cluster Profiles ===")
print(cluster_summary)

# ---- Save model + scaler + data for app ----
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
season_stats.to_csv('season_stats_with_clusters.csv', index=False)
cluster_summary.to_csv('cluster_summary.csv')

print("\nModel, scaler, and processed data saved!")