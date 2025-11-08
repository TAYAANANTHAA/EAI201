# features.py - creates extra features for FIFA model
import pandas as pd

def create_basic_features(df: pd.DataFrame):
    """
    Adds useful statistical features to FIFA data
    and returns dataframe with selected feature list.
    """
    df = df.copy()

    # Recalculate features if missing
    if "Goal_Difference" not in df.columns and "number of goals team1" in df.columns:
        df["Goal_Difference"] = df["number of goals team1"] - df["number of goals team2"]

    if "Total_Goals" not in df.columns and "number of goals team1" in df.columns:
        df["Total_Goals"] = df["number of goals team1"] + df["number of goals team2"]

    if "Total_Yellow_Cards" not in df.columns and "yellow cards team1" in df.columns:
        df["Total_Yellow_Cards"] = df["yellow cards team1"] + df["yellow cards team2"]

    if "Total_Red_Cards" not in df.columns and "red cards team1" in df.columns:
        df["Total_Red_Cards"] = df["red cards team1"] + df["red cards team2"]

    # Candidate features
    candidate_features = [
        "Goal_Difference",
        "Total_Goals",
        "Total_Yellow_Cards",
        "Total_Red_Cards"
    ]

    # Select only those present in df
    features = [f for f in candidate_features if f in df.columns]

    print("✅ Features ready:", features)
    return df, features
