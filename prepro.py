import re
import pandas as pd
import ftfy
import contractions
import nltk
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
nltk.download('stopwords')

# Load spaCy model once
nlp = spacy.load('en_core_web_sm')

def prepro(df: pd.DataFrame) -> pd.DataFrame:


    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # Get English stopwords
    stop_words = set(stopwords.words('english'))

    def clean_text(text: str) -> str:
        
        # Corrige mojibake y codificaciones rotas (p.ej. youâ€™ll → you'll)
        text = ftfy.fix_text(text)
        # Expande contracciones (you'll → you will)
        text = contractions.fix(text)
        text = text.replace('..',' ')
        # Elimina caracteres especiales y símbolos, manteniendo solo letras y espacios
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convierte a minúsculas
        text = text.lower()
        # Elimina espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        # Elimina espacios al inicio y final
        text = text.strip()
        # Elimina stopwords
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        return text

    def lemmatize_text(text: str) -> str:
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc])
    
    def tokenize_text(text: str) -> str:
        return text.split()
    
    def transform_label(label: str) -> bool:
        return label == 'yes'
    
    df['text_clean'] = df['full_text'].astype(str).apply(clean_text)
    df['text_clean'] = df['text_clean'].astype(str).apply(lemmatize_text)
    df['is_suicide'] = df['is_suicide'].astype(str).apply(transform_label)
    df['tokens'] = df['text_clean'].astype(str).apply(tokenize_text)
    df[['full_text', 'text_clean','tokens']].head(10)
    return df


