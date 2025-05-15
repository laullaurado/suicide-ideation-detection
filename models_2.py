from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class Model(Enum):
    DT = "DecisionTree"
    RF = "RandomForest"
    SVM = "SVM"
    LR = "LogisticRegression"
    XGB = "XGBoost"


def train_and_evaluate_model(X, y, model: Model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=80)

    all_cm = np.zeros((2,2), int)
    auc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model == Model.DT:
            clf = DecisionTreeClassifier(max_depth=8, random_state=80, class_weight='balanced')
        elif model == Model.RF:
            clf = RandomForestClassifier(max_depth=15, random_state=100, class_weight='balanced')
        elif model == Model.SVM:
            clf = SVC(probability=True, random_state=100, class_weight='balanced')
        elif model == Model.LR:
            clf = LogisticRegression(max_iter=1000, random_state=80, class_weight='balanced')
        elif model == Model.XGB:
            clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=80
            )

        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:,1]
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        all_cm += cm

        auc_scores.append(roc_auc_score(y_test, y_proba))
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        print(f"\nFold {fold}")
        print(f"AUC:      {auc_scores[-1]:.2f}")
        print(f"Accuracy: {accuracy_scores[-1]:.2f}")
        print(f"Precision:{precision_scores[-1]:.2f}")
        print(f"Recall:   {recall_scores[-1]:.2f}")
        print(f"F1-score: {f1_scores[-1]:.2f}")

    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=all_cm,
    #     display_labels=['no','yes']
    # )
    # disp.plot(cmap=plt.cm.get_cmap('Blues'))
    # plt.title("Accumulated Confusion Matrix (5 folds)")
    # plt.show()

    print("\n=== Average Metrics ===")
    print(f"AUC mean:      {np.mean(auc_scores):.2f}")
    print(f"Accuracy mean: {np.mean(accuracy_scores):.2f}")
    print(f"Precision mean:{np.mean(precision_scores):.2f}")
    print(f"Recall mean:   {np.mean(recall_scores):.2f}")
    print(f"F1-score mean: {np.mean(f1_scores):.2f}")

    tn, fp, fn, tp = all_cm.ravel()
    print("\n=== Cumulative Counts ===")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
