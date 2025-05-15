"""
Authors:
    - Lauren Lissette Llauradó Reyes
    - Carlos Alberto Sánchez Calderón
Date:
    2025-05-14
Description:
    This script defines a machine learning pipeline for suicide ideation detection.
    It includes a function to train and evaluate models.
"""

from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from sklearn.svm import SVC  # type: ignore
import xgboost as xgb  # type: ignore
from enum import Enum


class Model(Enum):
    DT = "DecisionTree"
    RF = "RandomForest"
    SVM = "SVM"
    LR = "LogisticRegression"
    XGB = "XGBoost"


def train_and_evaluate_model(X_tfidf, y, model: Model):
    """
    Trains and evaluates a machine learning model using Stratified K-Fold cross-validation.
    For each fold, it trains the specified model, evaluates it, and computes metrics such as:
        - AUC (Area Under the Curve)
        - Accuracy
        - Precision
        - Recall
        - F1-score
    The function also accumulates a confusion matrix across all folds and displays it.
    Args:
        X_tfidf(array-like or sparse matrix):
            The feature matrix (e.g., TF-IDF transformed data) used for training and testing.
        y(array-like):
            The target labels corresponding to the feature matrix.
        model(Model):
            An enumeration representing the type of model to train. 
            Supported models include:
            - Model.DT: Decision Tree
            - Model.RF: Random Forest
            - Model.SVM: Support Vector Machine
            - Model.LR: Logistic Regression
            - Model.XGB: XGBoost
    Returns:
        clf(sklearn classifier):
            The trained classifier from the last fold of cross-validation.
    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=80)

    all_cm = np.zeros((2, 2), int)
    auc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    match model:
        case Model.DT:
            print("\n=== Training Decision Tree ===")
            clf = DecisionTreeClassifier(
                max_depth=8, random_state=80, class_weight='balanced')
        case Model.RF:
            print("\n=== Training Random Forest ===")
            clf = RandomForestClassifier(
                max_depth=18, random_state=100, class_weight='balanced')
        case Model.SVM:
            print("\n=== Training SVM ===")
            clf = SVC(kernel='rbf', probability=True, random_state=42,
                      class_weight='balanced')
        case Model.LR:
            print("\n=== Training Logistic Regression ===")
            clf = LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced')
        case Model.XGB:
            print("\n=== Training XGBoost ===")
            clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=80)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tfidf, y), 1):
        X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        all_cm += cm

        auc_scores.append(roc_auc_score(y_test, y_proba))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        # print(f"\nFold {fold}")
        # print(f"AUC:      {auc_scores[-1]:.2f}")
        # print(f"Accuracy: {accuracy_scores[-1]:.2f}")
        # print(f"Precision:{precision_scores[-1]:.2f}")
        # print(f"Recall:   {recall_scores[-1]:.2f}")
        # print(f"F1-score: {f1_scores[-1]:.2f}")

    print("\n=== Metrics means ===")
    print(f"AUC mean:      {np.mean(auc_scores):.2f}")
    print(f"Accuracy mean: {np.mean(accuracy_scores):.2f}")
    print(f"Precision mean:{np.mean(precision_scores):.2f}")
    print(f"Recall mean:   {np.mean(recall_scores):.2f}")
    print(f"F1-score mean: {np.mean(f1_scores):.2f}")

    tn, fp, fn, tp = all_cm.ravel()
    print("\n=== Cumulative Counts (5 folds) ===")
    print(f"True Positives (TP): {int(tp)}")
    print(f"True Negatives (TN): {int(tn)}")
    print(f"False Positives (FP): {int(fp)}")
    print(f"False Negatives (FN): {int(fn)}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=all_cm, display_labels=['no', 'yes'])
    disp.plot(cmap=plt.colormaps["Blues"])
    plt.title("Cumulative confusion matrix (5 folds)")
    plt.show()

    return clf


def evaluate_model(clf, X_test, y_test):
    """
    Evaluates a trained model on a test set and computes various metrics.
    Args:
        clf(sklearn classifier):
            The trained classifier to evaluate.
        X_test(array-like or sparse matrix):
            The feature matrix for the test set.
        y_test(array-like):
            The true labels for the test set.
    Returns:
        None
    """
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['no', 'yes'])
    disp.plot(cmap=plt.colormaps["Blues"])
    plt.title("Confusion Matrix")
    plt.show()

    print("\n=== Test Results ===")
    print(f"AUC:      {roc_auc_score(y_test, y_proba):.2f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision:{precision_score(y_test, y_pred):.2f}")
    print(f"Recall:   {recall_score(y_test, y_pred):.2f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.2f}")

    tn, fp, fn, tp = cm.ravel()
    print("\n=== Cumulative Counts ===")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
