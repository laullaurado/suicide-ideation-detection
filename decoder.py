from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def create_tfidf_features(df: pd.DataFrame):

    # 2. Crea el vectorizador y ajústalo a tus datos
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df['text_clean'])

    # 3. Extrae tu variable objetivo
    y = df['is_suicide'].map({'no': 0, 'yes': 1}).values  # Convertir etiquetas a 0 y 1

    print("Matriz TF–IDF:", X_tfidf.shape)  # (n_muestras, n_términos)
    print("Número de características:", len(vectorizer.get_feature_names_out()))
    print("Primeras 10 características:", vectorizer.get_feature_names_out()[:10])
    
    return X_tfidf

