# cleaner.py - basic cleaning and preparation for FIFA dataset
import pandas as pd
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
OUT_PATH = DATA_DIR / "cleaned_fifa_dataset.csv"

# Edit these names if your CSV uses slightly different headers
COLS = {
    "team1_goals": "number of goals team1",
    "team2_goals": "number of goals team2",
    "yc1": "yellow cards team1",
    "yc2": "yellow cards team2",
    "rc1": "red cards team1",
    "rc2": "red cards team2",
    "team1": "team1",
    "team2": "team2",
    "date": "date"
}

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and engineers features in the FIFA dataset."""
    df = df.copy()

    # Drop duplicates if possible
    if all(col in df.columns for col in [COLS["team1"], COLS["team2"], COLS["date"]]):
        df = df.drop_duplicates(subset=[COLS["team1"], COLS["team2"], COLS["date"]])

    # Fill missing numeric values with median
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].fillna(df[c].median())

    # Create goal-based features
    if COLS["team1_goals"] in df.columns and COLS["team2_goals"] in df.columns:
        df["Goal_Difference"] = df[COLS["team1_goals"]] - df[COLS["team2_goals"]]
        df["Total_Goals"] = df[COLS["team1_goals"]] + df[COLS["team2_goals"]]

    # Discipline features
    if COLS["yc1"] in df.columns and COLS["yc2"] in df.columns:
        df["Total_Yellow_Cards"] = df[COLS["yc1"]].fillna(0) + df[COLS["yc2"]].fillna(0)
    if COLS["rc1"] in df.columns and COLS["rc2"] in df.columns:
        df["Total_Red_Cards"] = df[COLS["rc1"]].fillna(0) + df[COLS["rc2"]].fillna(0)

    # Create Match_Result column
    if "Match_Result" not in df.columns:
        def result(row):
            if row[COLS["team1_goals"]] > row[COLS["team2_goals"]]:
                return "Team1 Win"
            elif row[COLS["team1_goals"]] < row[COLS["team2_goals"]]:
                return "Team2 Win"
            else:
                return "Draw"
        df["Match_Result"] = df.apply(result, axis=1)

    # Save cleaned dataset
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Cleaned dataset saved to {OUT_PATH} (Rows: {len(df)})")
    return df


def load_and_clean(filename="Fifa_world_cup_matches.csv"):
    """Loads a CSV from the data folder and cleans it."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found!")
    df = pd.read_csv(path)
    return clean_df(df)
