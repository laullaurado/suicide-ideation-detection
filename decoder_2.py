import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        # ngram_range=(1,2),          # unigrams + bigrams
        sublinear_tf=True,          # logarithmic term-frequency scaling
        max_df=0.9,                 # drop very common terms
        min_df=5,                   # drop very rare terms
        max_features=20000,         # cap features
        norm='l2',
    )

def get_bow_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        ngram_range=(1,2),
        max_df=0.9,
        min_df=5,
        max_features=20000,
    )

def create_tfidf_features(df: pd.DataFrame):
    vect = get_tfidf_vectorizer()
    X = vect.fit_transform(df['text_clean'])
    y = df['is_suicide'].astype(int).values
    print("TF-IDF matrix shape:", X.shape)
    print("Number of features:", len(vect.get_feature_names_out()))
    print("First 10 features:", vect.get_feature_names_out()[:10])
    return X, vect, y


def create_bow_features(df: pd.DataFrame):
    vect = get_bow_vectorizer()
    X = vect.fit_transform(df['text_clean'])
    y = df['is_suicide'].astype(int).values
    print("BoW matrix shape:", X.shape)
    print("Number of features:", len(vect.get_feature_names_out()))
    print("First 10 features:", vect.get_feature_names_out()[:10])
    return X, vect, y