import re
import pandas as pd  # type: ignore
import ftfy  # type: ignore
import contractions  # type: ignore
import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore
import spacy  # type: ignore

# Download required NLTK data
nltk.download('stopwords')

# Load spaCy model once
nlp = spacy.load('en_core_web_sm')


def prepro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a DataFrame containing text data for suicide ideation detection.
    This function performs the following steps:
    1. Combines the 'title' and 'text' columns into a new column 'full_text'.
    2. Cleans the text data by:
        - Fixing mojibake and broken encodings.
        - Expanding contractions (e.g., "you'll" → "you will").
        - Removing special characters and symbols, keeping only letters and spaces.
        - Converting text to lowercase.
        - Removing multiple spaces and trimming leading/trailing spaces.
        - Removing English stopwords.
    3. Lemmatizes the cleaned text to reduce words to their base forms.
    4. Tokenizes the cleaned text into a list of words.
    5. Transforms the 'is_suicide' column into a boolean label (True for 'yes', False otherwise).
    Args:
        df (pd.DataFrame): Input DataFrame containing the text data. 
                           Expected columns: 'title', 'text', and 'is_suicide'.
    Returns:
        pd.DataFrame: The processed DataFrame with the following additional columns:
                      - 'full_text': Combined text from 'title' and 'text'.
                      - 'text_clean': Cleaned and lemmatized text.
                      - 'is_suicide': Boolean label indicating suicide ideation.
                      - 'tokens': Tokenized list of words from the cleaned text.
    """

    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # Get English stopwords
    stop_words = set(stopwords.words('english'))

    def clean_text(text: str) -> str:

        # Corrige mojibake y codificaciones rotas (p.ej. youâ€™ll → you'll)
        text = ftfy.fix_text(text)
        # Expande contracciones (you'll → you will)
        text = contractions.fix(text)
        text = text.replace('..', ' ')
        # Elimina caracteres especiales y símbolos, manteniendo solo letras y espacios
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convierte a minúsculas
        text = text.lower()
        # Elimina espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        # Elimina espacios al inicio y final
        text = text.strip()
        # Elimina stopwords
        text = ' '.join([word for word in text.split()
                        if word not in stop_words])

        return text

    def lemmatize_text(text: str) -> str:
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def tokenize_text(text: str) -> list[str]:
        return text.split()

    def transform_label(label: str) -> bool:
        return label == 'yes'

    df['text_clean'] = df['full_text'].astype(str).apply(clean_text)
    df['text_clean'] = df['text_clean'].astype(str).apply(lemmatize_text)
    df['is_suicide'] = df['is_suicide'].astype(str).apply(transform_label)
    df['tokens'] = df['text_clean'].astype(str).apply(tokenize_text)
    df[['full_text', 'text_clean', 'tokens']].head(10)
    return df
