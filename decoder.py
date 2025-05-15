from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
import pandas as pd  # type: ignore


def create_tfidf_features(df: pd.DataFrame):
    """
    Generates TF-IDF features from a given DataFrame and maps target labels.
    This function takes a DataFrame containing preprocessed text data and 
    computes the Term Frequency-Inverse Document Frequency (TF-IDF) matrix 
    for the text. It also maps the target labels ('is_suicide') to binary 
    values (0 for 'no', 1 for 'yes').
    Args:
        df (pd.DataFrame): A pandas DataFrame with at least two columns:
            - 'text_clean': Preprocessed text data.
            - 'is_suicide': Target labels indicating suicide ideation ('yes' or 'no').
    Returns:
        scipy.sparse.csr_matrix: The TF-IDF matrix representing the text data.
    """

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df['text_clean'])

    y = df['is_suicide'].map({'no': 0, 'yes': 1}).values

    print("Matriz TF-IDF:", X_tfidf.shape)
    print("Número de características:", len(
        vectorizer.get_feature_names_out()))
    print("Primeras 10 características:",
          vectorizer.get_feature_names_out()[:10])

    return X_tfidf


def create_bow_features(df: pd.DataFrame):
    """
    Generates Bag of Words (BoW) features from a given DataFrame and maps target labels.
    This function takes a DataFrame containing preprocessed text data and computes the 
    Bag of Words matrix for the text. It also maps the target labels ('is_suicide') 
    to binary values (0 for 'no', 1 for 'yes').
    Args:
        df (pd.DataFrame): A pandas DataFrame with at least two columns:
            - 'text_clean': Preprocessed text data.
            - 'is_suicide': Target labels indicating suicide ideation ('yes' or 'no').
    Returns:
        scipy.sparse.csr_matrix: The BoW matrix representing the text data.
    """
    from sklearn.feature_extraction.text import CountVectorizer  # type: ignore

    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(df['text_clean'])

    y = df['is_suicide'].map({'no': 0, 'yes': 1}).values

    print("Matriz BoW:", X_bow.shape)
    print("Número de características:", len(
        vectorizer.get_feature_names_out()))
    print("Primeras 10 características:",
          vectorizer.get_feature_names_out()[:10])

    return X_bow
