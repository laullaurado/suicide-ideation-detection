"""
Authors:
    - Lauren Lissette Llauradó Reyes
    - Carlos Alberto Sánchez Calderón
    - José Ángel Schiaffini Rodríguez
    - Karla Stefania Cruz Muñiz
Date:
    2025-05-14
Description:
    This module evaluates different combinations of text decoders (TF-IDF and BoW),
    dimensionality reduction (SVD), and classification models using 5-fold cross-validation
    to find the optimal configuration for suicide ideation detection.
"""

import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from time import time
from prepro import prepro
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.decomposition import TruncatedSVD  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.metrics import classification_report, roc_auc_score, roc_curve  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # type: ignore
from joblib import parallel_backend  # type: ignore


# Custom transformers that implement sklearn's interface
class TfidfTransformer(BaseEstimator, TransformerMixin):
    """Custom TF-IDF transformer that works with scikit-learn pipelines"""

    def __init__(self, max_df=0.9, min_df=5, max_features=20000, ngram_range=(1, 2),
                 sublinear_tf=True, stop_words='english', strip_accents='unicode'):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.stop_words = stop_words
        self.strip_accents = strip_accents
        self.vectorizer = None

    def fit(self, X, y=None):
        self.vectorizer = TfidfVectorizer(
            max_df=self.max_df,
            min_df=self.min_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words=self.stop_words,
            strip_accents=self.strip_accents
        )
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


class BowTransformer(BaseEstimator, TransformerMixin):
    """Custom Bag of Words transformer that works with scikit-learn pipelines"""

    def __init__(self, max_df=0.9, min_df=5, max_features=20000, ngram_range=(1, 1),
                 binary=False, stop_words='english'):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.stop_words = stop_words
        self.vectorizer = None

    def fit(self, X, y=None):
        self.vectorizer = CountVectorizer(
            max_df=self.max_df,
            min_df=self.min_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            binary=self.binary,
            stop_words=self.stop_words
        )
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


def get_tfidf_vectorizer():
    """Returns a configured TF-IDF transformer instance"""
    return TfidfTransformer()


def get_bow_vectorizer():
    """Returns a configured Bag of Words transformer instance"""
    return BowTransformer()


def evaluate_models():
    """
    Main function that evaluates different combinations of decoders, 
    dimensionality reduction techniques, and classifiers using 
    GridSearchCV with 5-fold cross-validation.
    """
    # Load and preprocess data
    print("Loading and preprocessing data...")
    start_time = time()
    df = pd.read_csv('./data_train(in).csv', encoding='latin-1')
    df = prepro(df)
    X_text = df['text_clean']
    y = df['is_suicide'].astype(int)
    print(f"Data loaded and preprocessed in {time() - start_time:.2f} seconds")

    # Base pipeline with placeholders
    pipe = Pipeline([
        ('vect', None),  # Placeholder for the vectorizer
        ('svd', None),   # Placeholder for dimensionality reduction
        ('clf', None)    # Placeholder for the classifier
    ])

    # Parameter grid for GridSearchCV
    param_grid = [
        # RandomForest with TF-IDF and BoW
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'vect__max_features': [10000, 20000],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'svd': [TruncatedSVD(random_state=42)],
            'svd__n_components': [100, 200],
            'clf': [RandomForestClassifier(class_weight='balanced', random_state=42)],
            'clf__n_estimators': [100, 300],
            'clf__max_depth': [None, 10, 15]
        },
        # LogisticRegression with TF-IDF and BoW
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'vect__max_features': [10000, 20000],
            'svd': [TruncatedSVD(random_state=42)],
            'svd__n_components': [100, 200],
            'clf': [LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)],
            'clf__C': [0.1, 1.0, 10.0]
        },
        # SVM with TF-IDF and BoW
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'vect__max_features': [10000, 20000],
            'svd': [TruncatedSVD(random_state=42)],
            'svd__n_components': [100, 200],
            'clf': [SVC(probability=True, class_weight='balanced', random_state=42)],
            'clf__kernel': ['linear', 'rbf'],
            'clf__C': [0.1, 1.0, 10.0]
        },
        # DecisionTree with TF-IDF and BoW
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'vect__max_features': [10000, 20000],
            'svd': [TruncatedSVD(random_state=42)],
            'svd__n_components': [100, 200],
            'clf': [DecisionTreeClassifier(class_weight='balanced', random_state=42)],
            'clf__max_depth': [5, 10, 15, 20]
        },
        # XGBoost with TF-IDF and BoW
        {
            'vect': [get_tfidf_vectorizer(), get_bow_vectorizer()],
            'vect__max_features': [10000, 20000],
            'svd': [TruncatedSVD(random_state=42)],
            'svd__n_components': [100, 200],
            'clf': [xgb.XGBClassifier(eval_metric='logloss', random_state=42)],
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 6, 10],
            'clf__learning_rate': [0.05, 0.1, 0.2]
        }
    ]

    # Create and run GridSearchCV
    print("Starting GridSearchCV with 5-fold cross-validation...")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    start_time = time()
    # Use threading backend to avoid ResourceTracker errors
    with parallel_backend('threading'):
        grid.fit(X_text, y)
    total_time = time() - start_time
    print(f"GridSearchCV completed in {total_time:.2f} seconds")

    # Print detailed results
    print("\n=== Best Model ===")
    print(f"Best AUC Score: {grid.best_score_:.4f}")
    print("Best Parameters:")
    for param, value in grid.best_params_.items():
        if param == 'vect':
            if isinstance(value, TfidfTransformer):
                print("  Decoder: TF-IDF")
            else:
                print("  Decoder: Bag of Words")
        elif param == 'svd':
            print(f"  SVD Components: {value.n_components}")
        elif param == 'clf':
            print(f"  Classifier: {type(value).__name__}")
        else:
            print(f"  {param}: {value}")

    # Results for all models
    print("\n=== Top 10 Models ===")
    results = pd.DataFrame(grid.cv_results_)
    results = results.sort_values('rank_test_score')

    for i, row in results.head(10).iterrows():
        params = row['params']
        decoder_type = "TF-IDF" if isinstance(
            params['vect'], TfidfTransformer) else "BoW"
        clf_type = type(params['clf']).__name__
        svd_components = params['svd'].n_components
        mean_score = row['mean_test_score']
        std_score = row['std_test_score']

        print(f"\nRank {row['rank_test_score']}")
        print(f"Decoder: {decoder_type}")
        print(f"SVD Components: {svd_components}")
        print(f"Classifier: {clf_type}")
        print(f"Mean AUC: {mean_score:.4f} (±{std_score:.4f})")

    # Save the best model's results
    best_model = grid.best_estimator_
    y_proba = best_model.predict_proba(X_text)[:, 1]

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (AUC = {roc_auc_score(y, y_proba):.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Model')
    plt.legend(loc="lower right")
    plt.savefig('best_model_roc.png')
    plt.close()

    # Evaluate on the entire dataset
    y_pred = best_model.predict(X_text)
    print("\n=== Classification Report (Best Model) ===")
    print(classification_report(y, y_pred))

    return grid, best_model


if __name__ == "__main__":
    grid, best_model = evaluate_models()

    # Save best model if needed
    # import joblib
    # joblib.dump(best_model, 'best_model.joblib')
