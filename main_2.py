import pandas as pd
from prepro import prepro
from decoder_2 import create_tfidf_features, create_bow_features, get_tfidf_vectorizer
from models_2 import train_and_evaluate_model, Model
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. Load and preprocess data
df = pd.read_csv('./data_train(in).csv', encoding='latin-1')
df = prepro(df)

# 2. Create features
X_tfidf, tfidf_vect, y = create_tfidf_features(df)
X_bow, bow_vect, y    = create_bow_features(df)

# 3. Train & evaluate with default hyperparameters
train_and_evaluate_model(X_tfidf, y, Model.RF)
train_and_evaluate_model(X_bow, y,    Model.SVM)

# 4. Example: Grid Search over RF + TF-IDF + SVD
pipeline = Pipeline([
    ('tfidf', get_tfidf_vectorizer()),
    ('svd', TruncatedSVD(random_state=80)),
    ('clf', RandomForestClassifier(random_state=80))
])
param_grid = {
    'tfidf__ngram_range': [(1,1),(1,2)],
    'svd__n_components':   [100,200],
    'clf__n_estimators':   [100,300],
    'clf__max_depth':      [None,10,20],
    'clf__class_weight':   ['balanced', None]
}
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid.fit(df['text_clean'], y)
print("Grid best AUC:", grid.best_score_)
print("Grid best params:", grid.best_params_)
