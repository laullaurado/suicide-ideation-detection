"""
Zero-shot classification using BART-large-mnli model.
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import torch
from tqdm import tqdm


def run_zero_shot_classification(df, model_name, batch_size=16, device=None):
    """
    Run zero-shot classification on text data using a transformer model.

    Args:
        df (pd.DataFrame): DataFrame with text data
        model_name (str): Name of the model to use
        batch_size (int): Batch size for inference
        device (int or str, optional): Device to run on (None for auto, -1 for CPU, 0+ for specific GPU)

    Returns:
        dict: Dictionary with results
    """
    print(f"\n{'='*50}")
    print(f"Running zero-shot classification with {model_name}")
    print(f"{'='*50}")

    # Set device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # Prepare input text
    print("Preparing input text...")
    df["input_text"] = df["title"].fillna("").astype(
        str).str.strip() + " " + df["text"].fillna("").astype(str).str.strip()

    # Map labels to binary
    if 'is_suicide' in df.columns:
        if df['is_suicide'].dtype == 'object':
            df["actual_binary"] = df["is_suicide"].map({"yes": 1, "no": 0})
        else:
            df["actual_binary"] = df["is_suicide"]

    # Initialize zero-shot classification pipeline
    print(
        f"Initializing zero-shot classification pipeline on device {device}...")
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )

    # Process in batches
    print(f"Running inference with batch size {batch_size}...")
    texts = df["input_text"].tolist()

    all_results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:min(i + batch_size, len(texts))]
        batch_results = classifier(
            sequences=batch_texts,
            candidate_labels=["suicide", "not suicide"],
            hypothesis_template="This text is about {}.",
        )
        all_results.extend(batch_results)

    # Extract predictions
    predicted_labels = []
    predicted_scores = []

    for result in all_results:
        scores = result['scores']
        labels = result['labels']

        suicide_idx = labels.index("suicide")
        suicide_score = scores[suicide_idx]

        predicted_scores.append(suicide_score)
        predicted_labels.append(1 if suicide_score > 0.5 else 0)

    # Store predictions
    df["zero_shot_score"] = predicted_scores
    df["zero_shot_pred"] = predicted_labels

    # Evaluate if we have ground truth
    metrics = {}
    if "actual_binary" in df.columns:
        y_true = df["actual_binary"].values
        y_pred = df["zero_shot_pred"].values
        y_score = df["zero_shot_score"].values

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_score),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score
        }

        print("\n=== Zero-Shot Classification Results ===")
        print(f"Model: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    return {
        "model_name": f"zero-shot-{model_name.split('/')[-1]}",
        "df": df,
        "metrics": metrics
    }
