# scraper.py - loads local CSV (safe). Template for scraping if needed.
import os
import pandas as pd

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def load_local(filename="Fifa_world_cup_matches.csv"):
    """Load a local FIFA dataset from the data folder."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local file not found: {path}")
    df = pd.read_csv(path)
    print("✅ Loaded local dataset:", path, "Shape:", df.shape)
    return df
