# modeler.py - train and save RandomForest model
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Folder to save model
BASE = Path(__file__).resolve().parents[1]
MODELS = BASE / "outputs" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def train_rf(df, features, target="Match_Result", save_name="rf_model.pkl"):
    """
    Trains a Random Forest classifier using selected features and saves the model.
    """
    X = df[features]
    y = df[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)

    model_package = {"model": rf, "le": le, "features": features}
    with open(MODELS / save_name, "wb") as f:
        pickle.dump(model_package, f)

    print(f"✅ Model trained and saved at {MODELS / save_name}")
    return rf, le, X_test, y_test

def load_model(path=None):
    """Load a saved model package."""
    if path is None:
        path = MODELS / "rf_model.pkl"
    with open(path, "rb") as f:
        pkg = pickle.load(f)
    return pkg["model"], pkg["le"], pkg["features"]

def predict(model, le, features, df):
    """Make predictions with a trained model."""
    X = df[features]
    preds = model.predict(X)
    probs = None
    try:
        probs = model.predict_proba(X)
    except Exception:
        pass
    labels = le.inverse_transform(preds)
    return labels, probs
