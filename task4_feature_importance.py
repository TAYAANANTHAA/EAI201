 ----------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("cleaned_fifa_dataset.csv")
print("✅ Data Loaded:", df.shape)

# Select features and target
X = df[["Goal_Difference", "Total_Goals", "Total_Yellow_Cards", "Total_Red_Cards"]]
y = df["Match_Result"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# --- Logistic Regression ---
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_scaled, y_train)

# Coefficients importance
log_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(log_model.coef_[0])
}).sort_values(by='Importance', ascending=False)
print("\n📊 Logistic Regression Feature Importance:\n", log_importance)

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\n🌲 Random Forest Feature Importance:\n", rf_importance)

# --- Plot ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.barplot(x="Importance", y="Feature", data=log_importance, color="blue")
plt.title("Logistic Regression - Feature Importance")

plt.subplot(1, 2, 2)
sns.barplot(x="Importance", y="Feature", data=rf_importance, color="green")
plt.title("Random Forest - Feature Importance")

plt.tight_layout()
plt.show()
