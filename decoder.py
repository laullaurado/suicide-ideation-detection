"""
Authors:
    - Lauren Lissette Llauradó Reyes
    - José Ángel Schiaffini Rodríguez
Date:
    2025-05-14
Description:
    This module contains functions and classes for feature extraction and dimensionality reduction.
    It includes functions to create TF-IDF and Bag of Words features from text data, 
    as well as a function for performing randomized Singular Value Decomposition (SVD) on the features.
"""


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # type: ignore
import pandas as pd  # type: ignore
from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore
from sklearn.utils.extmath import randomized_svd  # type: ignore


def create_tfidf_features(df: pd.DataFrame):
    """
    Generates TF-IDF features from a given DataFrame and maps target labels.
    This function takes a DataFrame containing preprocessed text data and 
    computes the Term Frequency-Inverse Document Frequency (TF-IDF) matrix 
    for the text.
    Args:
        df (pd.DataFrame): A pandas DataFrame with at least two columns:
            - 'text_clean': Preprocessed text data.
            - 'is_suicide': Target labels indicating suicide ideation ('yes' or 'no').
    Returns:
        scipy.sparse.csr_matrix: The TF-IDF matrix representing the text data.
    """

    vectorizer = TfidfVectorizer(ngram_range=(
        1, 3), sublinear_tf=True, max_df=0.9, min_df=5, max_features=10000)
    X_tfidf = vectorizer.fit_transform(df['text_clean'])

    # print("TF-IDF matrix:", X_tfidf.shape)
    # print("No. of features:", len(
    #     vectorizer.get_feature_names_out()))
    # print("First 10 features:",
    #       vectorizer.get_feature_names_out()[:10])

    return X_tfidf


def create_bow_features(df: pd.DataFrame):
    """
    Generates Bag of Words (BoW) features from a given DataFrame and maps target labels.
    This function takes a DataFrame containing preprocessed text data and computes the 
    Bag of Words matrix for the text.
    Args:
        df (pd.DataFrame): A pandas DataFrame with at least two columns:
            - 'text_clean': Preprocessed text data.
            - 'is_suicide': Target labels indicating suicide ideation ('yes' or 'no').
    Returns:
        scipy.sparse.csr_matrix: The BoW matrix representing the text data.
    """

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    X_bow = vectorizer.fit_transform(df['text_clean'])

    y = df['is_suicide'].map({'no': 0, 'yes': 1}).values

    # print("BoW matrix:", X_bow.shape)
    # print("No. of features:", len(
    #     vectorizer.get_feature_names_out()))
    # print("First 10 features:",
    #       vectorizer.get_feature_names_out()[:10])

    return X_bow


def randomized_svd_transformer(X, n_components=100, n_iter=5, random_state=None):
    """
    Applies a randomized Singular Value Decomposition (SVD) transformation to the input matrix.
    Args:
        X (array-like or sparse matrix, shape (n_samples, n_features)): 
            The input data matrix to be transformed. 
        n_components (int, optional):
            The number of singular values and vectors to compute. Defaults to 200.
        n_iter (int, optional):
            The number of iterations for the randomized SVD solver. Defaults to 5.
        random_state (int, RandomState instance, or None, optional): 
            The seed or random number generator for reproducibility. Defaults to None.
    Returns:
        ndarray (shape (n_samples, n_components)):
            The transformed data matrix after applying the randomized SVD.
    """

    _, _, VT = randomized_svd(
        M=X, n_components=n_components, n_iter=n_iter, random_state=random_state)
    components_ = VT
    return X.dot(components_.T)
