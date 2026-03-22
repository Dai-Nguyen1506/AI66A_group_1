from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    average_precision_score, precision_recall_curve
    )
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n🔹 {name}")
    print("-" * 40)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC:", average_precision_score(y_test, y_prob))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def evaluate_threshold(model, X_test, y_test, threshold=0.3, name="Model"):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    print(f"\n🔹 {name} (Threshold = {threshold})")
    print("-" * 40)
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC:", average_precision_score(y_test, y_prob))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def plot_PR_curve(model, X_test, y_test, name="Model"):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend()
    plt.grid()
    plt.show()