"""
Configuration settings for suicide ideation detection system.
"""

# Data paths
TRAIN_DATA_PATH = './data_train(in).csv'
TEST_DATA_PATH = './data_test_fold2(in).csv'

# Random seed for reproducibility
RANDOM_SEED = 42

# Embedding settings
EMBEDDING_CONFIGS = {
    'tfidf': {
        'max_features': 20000,
        'ngram_range': (1, 2),
        'max_df': 0.9,
        'min_df': 5
    },
    'bow': {
        'max_features': 20000,
        'ngram_range': (1, 1),
        'max_df': 0.9,
        'min_df': 5
    },
    'word2vec': {
        'vector_size': 300,
        'window': 5,
        'min_count': 1,
        'workers': 4,
        'sg': 1  # Skip-gram
    },
    'gpt2': {
        'model_name': 'gpt2-medium',
        'max_length': 512,
        'batch_size': 4
    }
}

# Cross-validation settings
CV_FOLDS = 5

# LLM settings
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
ZERO_SHOT_BATCH_SIZE = 16
FEW_SHOT_MODELS = ["deepseek-r1", "llama3"]
OLLAMA_URL = "http://localhost:11434"
