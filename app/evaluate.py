# evaluate.py - evaluates the Random Forest and Logistic Regression models
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import numpy as np
from pathlib import Path

PLOTS = Path(__file__).resolve().parents[1] / "outputs" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def evaluate_model(model, X_test, y_test, le, name="Random Forest"):
    """Evaluate model and save plots."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 {name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(PLOTS / f"{name}_confusion_matrix.png")
    plt.close()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(le.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        for i, label in enumerate(le.classes_):
            plt.plot(fpr[i], tpr[i], label=f"{label} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(PLOTS / f"{name}_ROC.png")
        plt.close()

    print(f"✅ Confusion matrix and ROC saved to: {PLOTS}")
