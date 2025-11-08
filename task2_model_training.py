# --------------------------------------------------
# Task 5: Final Prediction and Reflection
# Author: V.V. Tayaananthaa (24UG00548)
# --------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- 1️⃣ Load cleaned dataset ---
df = pd.read_csv(r"C:\Users\Huawei\Documents\FIFAPROJECT\fifa_clean_and_scrape\cleaned_fifa_dataset.csv")
print("✅ Data Loaded Successfully:", df.shape)

# --- 2️⃣ Select features and target column ---
features = ["Goal_Difference", "Total_Goals", "Total_Yellow_Cards", "Total_Red_Cards"]
target = "Match_Result"

X = df[features]
y = df[target]

# --- 3️⃣ Encode target labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- 4️⃣ Split into training & testing sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- 5️⃣ Train the best performing model (Random Forest) ---
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# --- 6️⃣ Evaluate performance on test data ---
y_pred = rf.predict(X_test)
print("\n--- Model Evaluation on Test Data ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- 7️⃣ Simulate 2026 World Cup prediction ---
# In real use, this would be new data. For demo, we use the same dataset.
df["Predicted_Result"] = rf.predict(X)

# Count predicted wins per team
team1_wins = df[df["Predicted_Result"] == le.transform(["Team1 Win"])[0]]["Team1"].value_counts()
team2_wins = df[df["Predicted_Result"] == le.transform(["Team2 Win"])[0]]["Team2"].value_counts()

# Combine and rank teams by total predicted wins
total_wins = (team1_wins.add(team2_wins, fill_value=0)).sort_values(ascending=False)

# --- 8️⃣ Display top 2 finalists ---
finalists = total_wins.head(2).index.tolist()
print("\n🏆 Predicted FIFA 2026 Finalists:")
for i, team in enumerate(finalists, start=1):
    print(f"{i}. {team}")

# --- 9️⃣ Reflection Notes (for report) ---
"""
🧠 Reflection and Discussion
----------------------------
• Model Used: Random Forest Classifier (best performance: 100% accuracy on test data)
• Key Features Influencing Predictions:
  - Goal Difference → Strong indicator of dominance in matches.
  - Total Goals → Reflects attacking performance.
  - Yellow/Red Cards → Discipline factors that may impact success.
• Limitations:
  - Dataset may not include future match dynamics or player conditions.
  - Model trained on past data — real sports outcomes have randomness.
  - A perfect accuracy might mean overfitting due to limited dataset size.
• Ethical Considerations:
  - Machine learning predictions in sports should complement, not replace, expert judgment.
  - Responsible presentation is needed — avoid misleading fans or media hype.
• Conclusion:
  - Random Forest predicts likely finalists for FIFA 2026 as above.
  - Future updates can include real qualifier data and player-level metrics for improved accuracy.
"""

print("\n✅ Final Prediction and Reflection Completed Successfully!")
import pickle
import os

model_path = r"C:\Users\Huawei\Documents\FIFAPROJECT\outputs\models"
os.makedirs(model_path, exist_ok=True)

# Save Random Forest model
with open(os.path.join(model_path, "rf_model.pkl"), "wb") as f:
    pickle.dump(rf, f)

print("✅ Model saved successfully at:", os.path.join(model_path, "rf_model.pkl"))

