"""
Description:
    Text preprocessing functions for suicide ideation detection.
"""

import re
import pandas as pd
import ftfy
import contractions
import nltk
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Load spaCy model once
nlp = spacy.load('en_core_web_sm')


def clean_text(text: str) -> str:
    """Cleans and preprocesses a text string."""
    stop_words = set(stopwords.words('english'))

    # Fix mojibake and broken encodings
    text = ftfy.fix_text(text)
    # Expand contractions
    text = contractions.fix(text)
    # Remove special characters and symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing spaces
    text = text.strip()
    # Remove English stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text


def lemmatize_text(text: str) -> str:
    """Lemmatizes the input text."""
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])


def tokenize_text(text: str) -> list[str]:
    """Splits the given text string into individual words."""
    return text.split()


def transform_label(label: str) -> int:
    """Converts a textual label into a numerical representation."""
    return 1 if label == 'yes' else 0


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses a DataFrame containing text data."""
    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['text_clean'] = df['full_text'].astype(str).apply(clean_text)
    df['text_clean'] = df['text_clean'].astype(str).apply(lemmatize_text)
    df['is_suicide'] = df['is_suicide'].astype(str).apply(transform_label)
    df['tokens'] = df['text_clean'].astype(str).apply(tokenize_text)
    return df
