import streamlit as st
import pandas as pd
import numpy as np
import random

# ---- PAGE CONFIG ----
st.set_page_config(page_title="FIFA 2026 Predictor", page_icon="⚽", layout="wide")

# ---- BACKGROUND STYLING ----
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1508253578933-20b9e29a0e2e?auto=format&fit=crop&w=1950&q=80");
background-size: cover;
background-position: center;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
div.stButton > button {
background-color: #00A36C;
color: white;
font-size: 16px;
border-radius: 10px;
}
.card {
background-color: rgba(0,0,0,0.7);
color: white;
padding: 25px;
border-radius: 15px;
text-align: center;
box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h1 style='text-align:center; color:white;'>🏆 FIFA 2026 Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gold;'>By V.V. Tayaananthaa | Chanakya University</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---- LOAD DATA ----
try:
    df = pd.read_csv(r"C:\Users\Huawei\Documents\FIFAPROJECT\data\Fifa_world_cup_matches.csv")
    st.success("✅ Dataset loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading dataset: {e}")
    df = None

if df is not None:
    teams = sorted(list(set(df["team1"].unique()) | set(df["team2"].unique())))

    # ---- SECTION 1: 2026 FINALISTS PREDICTION ----
    st.markdown("<h2 style='color:gold; text-align:center;'>🏅 Predict FIFA 2026 Finalists</h2>", unsafe_allow_html=True)
    if st.button("🎯 Predict 2026 Finalists"):
        # Simulate model predicting top 2 teams (replace with real model later)
        team_scores = {team: random.randint(70, 100) for team in teams}
        top2 = sorted(team_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        finalist1, finalist2 = top2[0][0], top2[1][0]

        st.markdown("<h4 style='text-align:center; color:white;'>Predicted 2026 Finalists:</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='card'><h2>{finalist1}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='card'><h2>{finalist2}</h2></div>", unsafe_allow_html=True)

        # Predict Champion
        winner = random.choice([finalist1, finalist2])
        st.markdown(
            f"<div class='card' style='background-color:gold;'><h2>🏆 Predicted Champion:</h2><h1>{winner}</h1></div>",
            unsafe_allow_html=True,
        )

        # Download Finalist Data
        result_df = pd.DataFrame({
            "Finalist 1": [finalist1],
            "Finalist 2": [finalist2],
            "Champion": [winner]
        })
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download 2026 Final Prediction",
            data=csv,
            file_name="FIFA_2026_Finalists.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ---- SECTION 2: MANUAL MATCH PREDICTOR ----
    st.markdown("<h2 style='color:white;'>⚽ Manual Match Prediction</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", teams)
    with col2:
        team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])

    if st.button("🔮 Predict Match Winner"):
        winner = random.choice([team1, team2, "Draw"])
        color = "lightgreen" if winner == team1 else "lightblue" if winner == team2 else "gold"
        st.markdown(
            f"<div class='card' style='background-color:{color};'><h2>🏅 Predicted Winner:</h2><h1>{winner}</h1></div>",
            unsafe_allow_html=True,
        )

# ---- FOOTER ----
st.markdown("---")
st.markdown("<p style='text-align:center; color:white;'>© 2025 FIFA Predictor | Designed by V.V. Tayaananthaa ⚽</p>", unsafe_allow_html=True)
