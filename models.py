from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


from enum import Enum

class Model(Enum):
    DT = "DecisionTree"
    RF = "RandomForest" 
    SVM = "SVM"
    LR = "LogisticRegression"

def train_and_evaluate_model(X_tfidf, y, model: Model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=80)

    # Acumuladores
    all_cm = np.zeros((2, 2), int)
    auc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tfidf, y), 1):
        X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 3) Entrenar el clasificador
        if model == Model.DT:
            print("Training Decision Tree")
            clf = DecisionTreeClassifier(max_depth=8, random_state=80)
        elif model == Model.RF:
            ...
        elif model == Model.SVM:
            print("Training SVM")
            clf = SVC()
        elif model == Model.LR:
            print("Training Logistic Regression")
            clf = LogisticRegression(max_iter=1000, random_state=80)
        clf.fit(X_train, y_train)

        # 4) Predecir y calcular probabilidades
        y_pred = clf.predict(X_test)
                

        # 5) Acumular la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        all_cm += cm

        # 6) Calcular métricas
        auc_scores.append(roc_auc_score(y_test, y_pred))
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

    # 7) Mostrar la matriz de confusión acumulada
    disp = ConfusionMatrixDisplay(confusion_matrix=all_cm, display_labels=['no', 'yes'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de confusión acumulada (5 folds)")
    plt.show()

    # 8) Resumen de métricas promedio
    print("\n=== Métricas promedio ===")
    print(f"AUC mean:      {np.mean(auc_scores):.2f}")
    print(f"Accuracy mean: {np.mean(accuracy_scores):.2f}")
    print(f"Precision mean:{np.mean(precision_scores):.2f}")
    print(f"Recall mean:   {np.mean(recall_scores):.2f}")
    print(f"F1-score mean: {np.mean(f1_scores):.2f}")

    # TP, TN, FP, FN
    tn, fp, fn, tp = all_cm.ravel()
    print("\nConteo acumulado (5 folds):")
    print(f"True Positives (TP): {int(tp)}")
    print(f"True Negatives (TN): {int(tn)}")
    print(f"False Positives (FP): {int(fp)}")
    print(f"False Negatives (FN): {int(fn)}")