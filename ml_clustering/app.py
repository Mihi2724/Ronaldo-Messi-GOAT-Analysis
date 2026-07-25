import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---- Load saved model, scaler, data (using full paths) ----
kmeans = joblib.load(os.path.join(BASE_DIR, 'kmeans_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
season_stats = pd.read_csv(os.path.join(BASE_DIR, 'season_stats_with_clusters.csv'))

cluster_labels = {
    0: "Prolific Right-Footed Finisher",
    1: "Lower-Output Season",
    2: "Left-Foot Dominant Playmaker"
}

st.title("⚽ GOAT Analysis: Messi vs Ronaldo")
st.write("Explore season-by-season playing style clusters, or enter hypothetical stats to see which style they match.")

# ---- Tab 1: Explore existing seasons ----
tab1, tab2 = st.tabs(["📊 Explore Seasons", "🔮 Predict Playing Style"])

with tab1:
    player_filter = st.selectbox("Select Player", ["Both", "Lionel Messi", "Cristiano Ronaldo"])
    
    if player_filter != "Both":
        display_data = season_stats[season_stats['Player'] == player_filter]
    else:
        display_data = season_stats

    display_data = display_data.copy()
    display_data['Style'] = display_data['Cluster'].map(cluster_labels)
    st.dataframe(display_data[['Player', 'Season', 'Goals', 'LeftFoot', 'RightFoot', 'Headers', 'Style']])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_map = {0: '#E74C3C', 1: '#3498DB', 2: '#2ECC71'}
    for player, marker in [('Lionel Messi', 'o'), ('Cristiano Ronaldo', 's')]:
        data = display_data[display_data['Player'] == player] if player_filter == "Both" else display_data
        for cluster in data['Cluster'].unique():
            subset = data[data['Cluster'] == cluster]
            ax.scatter(subset['Goals'], subset['RightFoot'], c=colors_map[cluster], marker=marker, s=100, alpha=0.8)
    ax.set_xlabel('Goals')
    ax.set_ylabel('Right-Foot Goals')
    ax.set_title('Season Clusters: Goals vs Right-Foot Goals')
    st.pyplot(fig)

with tab2:
    st.write("Enter hypothetical season stats to see which playing-style cluster it matches:")
    
    col1, col2 = st.columns(2)
    with col1:
        goals = st.number_input("Total Goals", min_value=0, value=40)
        penalties = st.number_input("Penalties", min_value=0, value=5)
        headers = st.number_input("Headers", min_value=0, value=5)
        freekicks = st.number_input("Free Kicks", min_value=0, value=2)
    with col2:
        leftfoot = st.number_input("Left-Foot Goals", min_value=0, value=10)
        rightfoot = st.number_input("Right-Foot Goals", min_value=0, value=15)
        awaygoals = st.number_input("Away Goals", min_value=0, value=18)

    if st.button("Predict Playing Style"):
        input_data = [[goals, penalties, headers, freekicks, leftfoot, rightfoot, awaygoals]]
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        st.success(f"This season profile matches: **{cluster_labels[cluster]}** (Cluster {cluster})")