# ---------------------------------------------------------
# TASK 3: MODEL EVALUATION AND VISUALIZATION
# Name: V.V. Tayaananthaa | Reg No: 24UG00548
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
)

# 1️⃣ Load Dataset
df = pd.read_csv("cleaned_fifa_dataset.csv")
print("✅ Data Loaded:", df.shape)

# 2️⃣ Feature Selection and Encoding
X = df[["Goal_Difference", "Total_Goals", "Total_Yellow_Cards", "Total_Red_Cards"]]
y = df["Match_Result"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3️⃣ Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4️⃣ Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Train Models
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(max_depth=3, n_estimators=50, random_state=42)

log_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train, y_train)

# 6️⃣ Predictions
log_pred = log_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

# 7️⃣ Evaluation Metrics
def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-Score": f1_score(y_true, y_pred, average="weighted")
    }

results = []
results.append(evaluate_model("Logistic Regression", y_test, log_pred))
results.append(evaluate_model("Random Forest", y_test, rf_pred))

results_df = pd.DataFrame(results)
print("\n🔹 Summary Table:")
print(results_df)

# 8️⃣ Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Random Forest Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 9️⃣ ROC Curves (for visual comparison)
try:
    RocCurveDisplay.from_estimator(log_model, X_test_scaled, y_test, name="Logistic Regression")
    RocCurveDisplay.from_estimator(rf_model, X_test, y_test, name="Random Forest")
    plt.title("ROC Curve Comparison")
    plt.show()
except Exception as e:
    print("ROC curve could not be plotted (multiclass issue):", e)

# 🔟 Save Evaluation Summary
results_df.to_csv("model_evaluation_summary.csv", index=False)
print("\n✅ Evaluation Completed Successfully! Results saved to model_evaluation_summary.csv")
# ---------------------------------------------------------
# TASK 3: MODEL EVALUATION AND VISUALIZATION
# Name: V.V. Tayaananthaa | Reg No: 24UG00548
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
)

# 1️⃣ Load Dataset
df = pd.read_csv("cleaned_fifa_dataset.csv")
print("✅ Data Loaded:", df.shape)

# 2️⃣ Feature Selection and Encoding
X = df[["Goal_Difference", "Total_Goals", "Total_Yellow_Cards", "Total_Red_Cards"]]
y = df["Match_Result"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3️⃣ Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4️⃣ Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Train Models
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(max_depth=3, n_estimators=50, random_state=42)

log_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train, y_train)

# 6️⃣ Predictions
log_pred = log_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

# 7️⃣ Evaluation Metrics
def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-Score": f1_score(y_true, y_pred, average="weighted")
    }

results = []
results.append(evaluate_model("Logistic Regression", y_test, log_pred))
results.append(evaluate_model("Random Forest", y_test, rf_pred))

results_df = pd.DataFrame(results)
print("\n🔹 Summary Table:")
print(results_df)

# 8️⃣ Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Random Forest Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 9️⃣ ROC Curves (for visual comparison)
try:
    RocCurveDisplay.from_estimator(log_model, X_test_scaled, y_test, name="Logistic Regression")
    RocCurveDisplay.from_estimator(rf_model, X_test, y_test, name="Random Forest")
    plt.title("ROC Curve Comparison")
    plt.show()
except Exception as e:
    print("ROC curve could not be plotted (multiclass issue):", e)

# 🔟 Save Evaluation Summary
results_df.to_csv("model_evaluation_summary.csv", index=False)
print("\n✅ Evaluation Completed Successfully! Results saved to model_evaluation_summary.csv")
