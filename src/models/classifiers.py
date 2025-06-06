"""
Description:
    Classification models and evaluation functions for suicide ideation detection.
"""

import numpy as np
from enum import Enum
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)


class Model(Enum):
    DT = "DecisionTree"
    RF = "RandomForest"
    SVM = "SVM"
    LR = "LogisticRegression"
    XGB = "XGBoost"


def get_classifier(model_type: Model, random_state=42):
    """Get a classifier instance based on the model type."""
    match model_type:
        case Model.DT:
            return DecisionTreeClassifier(
                max_depth=8,
                random_state=random_state,
                class_weight='balanced'
            )
        case Model.RF:
            return RandomForestClassifier(
                max_depth=18,
                random_state=random_state,
                class_weight='balanced'
            )
        case Model.SVM:
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=random_state,
                class_weight='balanced'
            )
        case Model.LR:
            return LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced'
            )
        case Model.XGB:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            )


def cross_validate(X, y, model_type: Model, n_splits=5, random_state=42):
    """
    Train and evaluate a model using k-fold cross-validation.
    Returns performance metrics and the trained model from the last fold.
    """
    print(
        f"\n=== Training {model_type.value} with {n_splits}-fold cross-validation ===")

    # Initialize k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    # Metrics storage
    all_cm = np.zeros((2, 2), int)
    auc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Get classifier
    clf = get_classifier(model_type, random_state)

    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the model
        clf.fit(X_train, y_train)

        # Make predictions
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        all_cm += cm

        auc_scores.append(roc_auc_score(y_test, y_proba))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    # Calculate mean metrics
    metrics = {
        'auc': np.mean(auc_scores),
        'accuracy': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1': np.mean(f1_scores),
        'confusion_matrix': all_cm
    }

    # Print metrics summary
    print("\n=== Cross-validation Results ===")
    print(f"AUC mean:      {metrics['auc']:.4f}")
    print(f"Accuracy mean: {metrics['accuracy']:.4f}")
    print(f"Precision mean:{metrics['precision']:.4f}")
    print(f"Recall mean:   {metrics['recall']:.4f}")
    print(f"F1-score mean: {metrics['f1']:.4f}")

    # Confusion matrix details
    tn, fp, fn, tp = all_cm.ravel()
    print("\n=== Cumulative Counts (all folds) ===")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    return clf, metrics


def evaluate_test_set(clf, X_test, y_test):
    """Evaluate model performance on a test set."""
    print("\n=== Evaluating on Test Set ===")

    # Make predictions
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create metrics dictionary
    metrics = {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_test': y_test,
        'y_true': y_test
    }

    # Print metrics
    print("\n=== Test Set Results ===")
    print(f"AUC:      {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision:{precision:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Confusion matrix details
    tn, fp, fn, tp = cm.ravel()
    print("\n=== Test Set Counts ===")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    return metrics
