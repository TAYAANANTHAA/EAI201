# cli_app.py - main command line application for FIFA project
import pandas as pd
from scraper import load_local
from cleaner import load_and_clean
from features import create_basic_features
from modeler import train_rf, load_model, predict
from evaluate import evaluate_model
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUTPUTS = BASE / "outputs"
PREDICTIONS = OUTPUTS / "predictions"
PREDICTIONS.mkdir(parents=True, exist_ok=True)

def main_menu():
    print("\n⚽ FIFA WORLD CUP PREDICTION APP ⚽")
    print("1️⃣  Load Data")
    print("2️⃣  Clean Data")
    print("3️⃣  Create Features")
    print("4️⃣  Train Model")
    print("5️⃣  Evaluate Model")
    print("6️⃣  Predict Finalists")
    print("7️⃣  Exit")

def main():
    df = None
    features = None
    model = None
    le = None
    X_test = None
    y_test = None

    while True:
        main_menu()
        choice = input("\n👉 Enter your choice (1-7): ").strip()

        if choice == "1":
            df = load_local()
            print(df.head())

        elif choice == "2":
            df = load_and_clean()
            print(df.head())

        elif choice == "3":
            if df is None:
                print("⚠️ Please load or clean data first.")
            else:
                df, features = create_basic_features(df)

        elif choice == "4":
            if df is None or features is None:
                print("⚠️ Please clean data and create features first.")
            else:
                model, le, X_test, y_test = train_rf(df, features)

        elif choice == "5":
            if model is None:
                model, le, features = load_model()
            evaluate_model(model, X_test, y_test, le)

        elif choice == "6":
            if model is None:
                model, le, features = load_model()
            df = pd.read_csv(DATA / "cleaned_fifa_dataset.csv")
            df, features = create_basic_features(df)
            labels, _ = predict(model, le, features, df)

            df["Predicted_Result"] = labels
            top_team1 = df[df["Predicted_Result"] == "Team1 Win"]["team1"].value_counts()
            top_team2 = df[df["Predicted_Result"] == "Team2 Win"]["team2"].value_counts()

            total_wins = (top_team1.add(top_team2, fill_value=0)).sort_values(ascending=False)
            finalists = total_wins.head(2).index.tolist()

            print("\n🏆 Predicted Finalists for FIFA 2026:")
            for i, team in enumerate(finalists, start=1):
                print(f"{i}. {team}")

            save_path = PREDICTIONS / "final_predictions.csv"
            df.to_csv(save_path, index=False)
            print(f"✅ Predictions saved to {save_path}")

        elif choice == "7":
            print("👋 Exiting... Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter a number between 1-7.")

if __name__ == "__main__":
    main()
