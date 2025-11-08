import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- 1️⃣ LOAD THE DATASET ---
df = pd.read_csv("Fifa_world_cup_matches.csv")
print("✅ Loaded dataset:", df.shape)

# --- 2️⃣ SELECT & RENAME MAIN COLUMNS FOR EASIER USE ---
df = df.rename(columns={
    "team1": "Team1",
    "team2": "Team2",
    "number of goals team1": "Goals_T1",
    "number of goals team2": "Goals_T2",
    "yellow cards team1": "Yellows_T1",
    "yellow cards team2": "Yellows_T2",
    "red cards team1": "Reds_T1",
    "red cards team2": "Reds_T2",
})

# --- 3️⃣ FEATURE ENGINEERING ---
df["Goal_Difference"] = df["Goals_T1"] - df["Goals_T2"]
df["Total_Goals"] = df["Goals_T1"] + df["Goals_T2"]
df["Total_Yellow_Cards"] = df["Yellows_T1"] + df["Yellows_T2"]
df["Total_Red_Cards"] = df["Reds_T1"] + df["Reds_T2"]

def match_result(row):
    if row["Goals_T1"] > row["Goals_T2"]:
        return "Team1 Win"
    elif row["Goals_T1"] < row["Goals_T2"]:
        return "Team2 Win"
    else:
        return "Draw"

df["Match_Result"] = df.apply(match_result, axis=1)

# --- 4️⃣ CLEAN THE DATA ---
df = df.dropna(subset=["Team1", "Team2"])
df.to_csv("cleaned_fifa_dataset.csv", index=False)
print("✅ Saved cleaned_fifa_dataset.csv with new features")

# --- 5️⃣ SCRAPE FIFA RANKINGS (Wikipedia) ---
url = "https://en.wikipedia.org/wiki/FIFA_Men%27s_World_Ranking"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

table = soup.find("table", {"class": "wikitable"})
rows = table.find_all("tr")

teams, ranks = [], []
for row in rows[1:21]:
    cols = row.find_all("td")
    if len(cols) >= 3:
        ranks.append(cols[0].get_text(strip=True))
        teams.append(cols[1].get_text(strip=True))

ranking_df = pd.DataFrame({"Rank": ranks, "Team": teams})
ranking_df.to_csv("fifa_rankings.csv", index=False)
print("✅ Scraped and saved fifa_rankings.csv (Top 20 teams)")

print("🎉 All done! Created cleaned_fifa_dataset.csv and fifa_rankings.csv.")
