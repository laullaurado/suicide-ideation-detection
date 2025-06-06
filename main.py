"""
Main script for suicide ideation detection evaluation.
This script runs traditional ML, zero-shot, and few-shot approaches
and compares their performance.
"""

from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, RANDOM_SEED,
    EMBEDDING_CONFIGS, ZERO_SHOT_MODEL,
    ZERO_SHOT_BATCH_SIZE, FEW_SHOT_MODELS, OLLAMA_URL
)
from src.utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, plot_all_roc_curves,
    plot_model_comparison, display_best_model_metrics
)
from src.llm.few_shot import run_few_shot_classification
from src.llm.zero_shot import run_zero_shot_classification
from src.models import Model, cross_validate, evaluate_test_set
from src.features import (
    TfidfEmbedding, BowEmbedding, Word2VecEmbedding, GPT2Embedding
)
from src.preprocessing import preprocess_data
import os
import pandas as pd
import numpy as np
from time import time
import joblib
import matplotlib.pyplot as plt
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import modules

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


def evaluate_traditional_ml():
    """Evaluate traditional ML approaches."""
    print("\n\n")
    print(f"{'='*80}")
    print("EVALUATING TRADITIONAL ML APPROACHES")
    print(f"{'='*80}")

    # Load and preprocess training data
    print("Loading and preprocessing training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH, encoding='latin-1')
    train_df = preprocess_data(train_df)
    train_y = train_df['is_suicide'].values

    # Load and preprocess test data
    print("Loading and preprocessing test data...")
    test_df = pd.read_csv(TEST_DATA_PATH, encoding='latin-1')
    test_df = preprocess_data(test_df)
    test_y = test_df['is_suicide'].values

    # Results storage
    results = []

    # Define embedding methods to test
    embedding_methods = {
        'tfidf': TfidfEmbedding(**EMBEDDING_CONFIGS['tfidf']),
        'bow': BowEmbedding(**EMBEDDING_CONFIGS['bow']),
    }

    # Add Word2Vec and GPT-2 if requested (they take more time)
    include_advanced = True
    if include_advanced:
        embedding_methods['word2vec'] = Word2VecEmbedding(
            **EMBEDDING_CONFIGS['word2vec'])
        if torch.cuda.is_available():
            embedding_methods['gpt2'] = GPT2Embedding(
                **EMBEDDING_CONFIGS['gpt2'])

    # Models to evaluate
    models_to_evaluate = [Model.SVM, Model.LR]  # Fast models first

    # Add more complex models if requested
    include_complex = True
    if include_complex:
        models_to_evaluate.extend([Model.RF, Model.XGB])

    # Evaluate combinations
    for embedding_name, embedding_method in embedding_methods.items():
        # Extract features from training data
        print(f"\n{'='*50}")
        print(f"Creating {embedding_name} features...")

        if embedding_name == 'word2vec':
            X_train = embedding_method.fit_transform(
                train_df['tokens'].tolist())
            X_test = embedding_method.transform(test_df['tokens'].tolist())
        elif embedding_name == 'gpt2':
            X_train = embedding_method.fit_transform(
                train_df['text_clean'].tolist())
            X_test = embedding_method.transform(test_df['text_clean'].tolist())
        else:
            X_train = embedding_method.fit_transform(
                train_df['text_clean'].tolist())
            X_test = embedding_method.transform(test_df['text_clean'].tolist())

        # Ensure features are numpy arrays
        if not isinstance(X_train, np.ndarray):
            X_train = X_train.toarray()
        if not isinstance(X_test, np.ndarray):
            X_test = X_test.toarray()

        # Evaluate all models with this embedding
        for model_type in models_to_evaluate:
            model_name = f"{embedding_name}-{model_type.value}"
            print(f"\n{'='*50}")
            print(f"EVALUATING: {model_name}")
            print(f"{'='*50}")

            # Train and validate on training data
            clf, cv_metrics = cross_validate(
                X_train, train_y, model_type, random_state=RANDOM_SEED)

            # Evaluate on test data
            test_metrics = evaluate_test_set(clf, X_test, test_y)

            # Store results
            results.append({
                'model_name': model_name,
                'metrics': test_metrics
            })

    return results


def evaluate_zero_shot_classification():
    """Evaluate zero-shot classification."""
    print("\n\n")
    print(f"{'='*80}")
    print("EVALUATING ZERO-SHOT CLASSIFICATION")
    print(f"{'='*80}")

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH, encoding='latin-1')

    # Run zero-shot classification
    result = run_zero_shot_classification(
        test_df,
        model_name=ZERO_SHOT_MODEL,
        batch_size=ZERO_SHOT_BATCH_SIZE,
        device=None  # Auto-detect GPU
    )

    return [result]  # Return as list to match other methods


def evaluate_few_shot_classification():
    """Evaluate few-shot classification."""
    print("\n\n")
    print(f"{'='*80}")
    print("EVALUATING FEW-SHOT CLASSIFICATION")
    print(f"{'='*80}")

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH, encoding='latin-1')

    # Results storage
    results = []

    # Run few-shot classification for each model
    for model_name in FEW_SHOT_MODELS:
        try:
            result = run_few_shot_classification(
                test_df,
                model_name=model_name,
                ollama_url=OLLAMA_URL
            )
            results.append(result)
        except Exception as e:
            print(
                f"Error running few-shot classification with {model_name}: {e}")
            # Continue with other models

    return results


def main():
    """Main function to run all evaluations."""
    print("Suicide Ideation Detection - Comprehensive Evaluation")
    print(f"Training data: {TRAIN_DATA_PATH}")
    print(f"Test data: {TEST_DATA_PATH}")

    # Storage for all results
    all_results = []

    # 1. Traditional ML
    try:
        ml_results = evaluate_traditional_ml()
        all_results.extend(ml_results)
    except Exception as e:
        print(f"Error in traditional ML evaluation: {e}")

    # 2. Zero-shot classification
    try:
        zs_results = evaluate_zero_shot_classification()
        all_results.extend(zs_results)
    except Exception as e:
        print(f"Error in zero-shot classification: {e}")

    # 3. Few-shot classification
    try:
        fs_results = evaluate_few_shot_classification()
        all_results.extend(fs_results)
    except Exception as e:
        print(f"Error in few-shot classification: {e}")

    # Visualize results
    print("\n\n")
    print(f"{'='*80}")
    print("VISUALIZATION AND COMPARISON")
    print(f"{'='*80}")

    # Plot all ROC curves
    plot_all_roc_curves(all_results, save_path='results/all_roc_curves.png')

    # Model comparison
    plot_model_comparison(all_results, metric='auc',
                          save_path='results/auc_comparison.png')
    plot_model_comparison(all_results, metric='accuracy',
                          save_path='results/accuracy_comparison.png')

    # Display best model details
    best_model = display_best_model_metrics(all_results)

    # Save results summary
    results_summary = []
    for result in all_results:
        metrics = result.get('metrics', {})
        if metrics:
            results_summary.append({
                'model': result.get('model_name', 'Unknown'),
                'auc': metrics.get('auc', 0),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0)
            })

    if results_summary:
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('auc', ascending=False)
        results_df.to_csv('results/all_models_comparison.csv', index=False)
        print("\nResults saved to 'results/all_models_comparison.csv'")


if __name__ == "__main__":
    main()
