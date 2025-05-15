# File: pipeline.py
import pandas as pd
from prepro_2 import prepro
from decoder_2 import get_tfidf_vectorizer, get_bow_vectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from joblib import parallel_backend  # Add this import



def main():
    # Load and preprocess data
    df = pd.read_csv('./data_train(in).csv', encoding='latin-1')
    df = prepro(df)
    X_text = df['text_clean']
    y = df['is_suicide'].astype(int)

    # Base pipeline with placeholders
    pipe = Pipeline([
        ('vect', get_tfidf_vectorizer()),
        ('svd', TruncatedSVD(random_state=0)),
        ('clf', RandomForestClassifier(random_state=0))
    ])

    # Parameter grid covering both TF-IDF & BoW, all models, and varying random_state
    param_grid = [
        # RandomForest
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'svd':  [TruncatedSVD(n_components=n, random_state=0) for n in (100,200)],
            'clf':  [RandomForestClassifier(class_weight='balanced')],
            'clf__n_estimators':   [100,300],
            'clf__max_depth':      [None, 5, 10, 15, 18],
            'clf__random_state':   [42,80,100]
        },
        # LogisticRegression
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'svd':  [TruncatedSVD(n_components=200, random_state=0)],
            'clf':  [LogisticRegression(max_iter=1000, class_weight='balanced')],
            'clf__C':             [0.1,1,10],
            'clf__random_state':  [42,80,100]
        },
        # SVM
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'svd':  [TruncatedSVD(n_components=200, random_state=0)],
            'clf':  [SVC(probability=True, class_weight='balanced')],
            'clf__kernel':        ['linear','poly', 'rbf'],
            'clf__C':             [0.1,1,10],
            'clf__random_state':  [42,80,100]
        },
        # DecisionTree
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'svd':  [TruncatedSVD(n_components=200, random_state=0)],
            'clf':  [DecisionTreeClassifier(class_weight='balanced')],
            'clf__max_depth':     [5,10,15,18,20],
            'clf__random_state':  [42,80,100]
        },
        # XGBoost
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'svd':  [TruncatedSVD(n_components=200, random_state=0)],
            'clf':  [xgb.XGBClassifier(eval_metric='logloss')],
            'clf__n_estimators':  [100,200],
            'clf__max_depth':     [3,5,6,10],
            'clf__learning_rate': [0.05,0.1],
            'clf__random_state':  [42,80,100]
        }
    ]

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )

    with parallel_backend('threading'):
        grid.fit(X_text, y)

    # Print only the best results
    print(f"Best AUC : {grid.best_score_:.4f}")
    print(f"Best Params: {grid.best_params_}")


if __name__ == '__main__':
    main()
