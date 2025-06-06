"""
Description:
    Text embedding methods for suicide ideation detection:
    - TF-IDF: Term Frequency-Inverse Document Frequency
    - BOW: Bag of Words
    - Word2Vec: Word embeddings
    - GPT-2: Transformer-based embeddings
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import torch
from gensim.models import Word2Vec
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class TfidfEmbedding:
    """TF-IDF vectorization for text data."""

    def __init__(self, max_features=20000, ngram_range=(1, 2), max_df=0.9, min_df=5):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=True,
            strip_accents='unicode',
            stop_words='english'
        )

    def fit_transform(self, texts):
        """Fit and transform texts to TF-IDF features."""
        print("Generating TF-IDF features...")
        X = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF matrix shape: {X.shape}")
        print(
            f"Number of features: {len(self.vectorizer.get_feature_names_out())}")
        return X

    def transform(self, texts):
        """Transform new texts to TF-IDF features."""
        return self.vectorizer.transform(texts)


class BowEmbedding:
    """Bag of Words vectorization for text data."""

    def __init__(self, max_features=20000, ngram_range=(1, 1), max_df=0.9, min_df=5):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            binary=False,
            strip_accents='unicode',
            stop_words='english'
        )

    def fit_transform(self, texts):
        """Fit and transform texts to BOW features."""
        print("Generating BOW features...")
        X = self.vectorizer.fit_transform(texts)
        print(f"BOW matrix shape: {X.shape}")
        print(
            f"Number of features: {len(self.vectorizer.get_feature_names_out())}")
        return X

    def transform(self, texts):
        """Transform new texts to BOW features."""
        return self.vectorizer.transform(texts)


class Word2VecEmbedding:
    """Word2Vec embeddings for text data."""

    def __init__(self, vector_size=300, window=5, min_count=1, workers=4, sg=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg  # 1 for skip-gram, 0 for CBOW
        self.model = None

    def fit_transform(self, tokenized_texts):
        """Fit Word2Vec model and transform tokenized texts to document embeddings."""
        print("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )

        print("Generating document embeddings...")
        return self._get_document_vectors(tokenized_texts)

    def transform(self, tokenized_texts):
        """Transform new tokenized texts to document embeddings."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit_transform first.")
        return self._get_document_vectors(tokenized_texts)

    def _get_document_vectors(self, tokenized_texts):
        """Convert tokenized texts to document vectors by averaging word vectors."""
        document_vectors = np.zeros((len(tokenized_texts), self.vector_size))

        for i, tokens in enumerate(tokenized_texts):
            valid_tokens = [
                token for token in tokens if token in self.model.wv]
            if valid_tokens:
                # Average the word vectors for a document
                vectors = [self.model.wv[token] for token in valid_tokens]
                document_vectors[i] = np.mean(vectors, axis=0)

        print(f"Word2Vec document matrix shape: {document_vectors.shape}")
        return document_vectors


class GPT2Embedding:
    """GPT-2 embeddings for text data."""

    def __init__(self, model_name="gpt2-medium", max_length=512, batch_size=8):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None

    def _initialize_model(self):
        """Initialize the GPT-2 tokenizer and model."""
        print(f"Loading {self.model_name} model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        # GPT-2 doesn't have a padding token by default
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2Model.from_pretrained(self.model_name)
        self.model.to(DEVICE)
        self.model.eval()  # Set to evaluation mode

    def fit_transform(self, texts):
        """Generate GPT-2 embeddings for texts."""
        if self.tokenizer is None or self.model is None:
            self._initialize_model()

        print("Generating GPT-2 embeddings...")
        return self._get_embeddings(texts)

    def transform(self, texts):
        """Generate GPT-2 embeddings for new texts."""
        if self.tokenizer is None or self.model is None:
            raise ValueError(
                "Model not initialized. Call fit_transform first.")
        return self._get_embeddings(texts)

    def _get_embeddings(self, texts):
        """Extract GPT-2 embeddings by averaging the last hidden state."""
        all_embeddings = []

        # Process texts in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting GPT-2 embeddings"):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize and prepare batch
            encodings = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].to(DEVICE)
            attention_mask = encodings['attention_mask'].to(DEVICE)

            # Get embeddings without gradient calculation
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Get last hidden state
                # [batch_size, seq_len, hidden_size]
                last_hidden_state = outputs.last_hidden_state

                # Create a mask to ignore padding tokens (expand mask to match hidden state dimensions)
                mask_expanded = attention_mask.unsqueeze(
                    -1).expand(last_hidden_state.size()).float()

                # Apply mask and calculate mean over sequence length
                sum_hidden = torch.sum(
                    last_hidden_state * mask_expanded, dim=1)
                seq_lengths = torch.clamp(
                    attention_mask.sum(dim=1, keepdim=True), min=1.0)
                embeddings = sum_hidden / seq_lengths

                all_embeddings.append(embeddings.cpu().numpy())

        # Combine all batches
        embeddings_matrix = np.vstack(all_embeddings)
        print(f"GPT-2 embeddings matrix shape: {embeddings_matrix.shape}")
        return embeddings_matrix
