"""
Few-shot classification using Ollama models.
"""

import pandas as pd
import numpy as np
import requests
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from tqdm import tqdm
import re

# Few-shot examples - carefully selected balanced examples
FEW_SHOT_EXAMPLES = [
    {
        "title": "feeling so overwhelmed and hopeless",
        "text": "i have been so depressed these past couple weeks ever since i got back from vacation normal life and reality has hit me and its terrible. I'm 19 i havent moved out yet i dont really have a sufficient job i have one but its not enough money for me to support myself. at least i have a car and I'm taking online classes so. i feel stuck in this stupid fucking town. empty and alone man i just want someone to love me dude but I'm ugly and fat and worthless and no one is ever going to see I'm worth the chance. its just one thing after another to worry about. why not end it now before i really fuck up and become miserable? i dont see the point in going on I'm just really down right now i dont know what to do )): all there is darkness idk how to get out of this hole i hate myself i hate my body i hate my life i dont think it will ever get better",
        "label": "suicide"
    },
    {
        "title": "I wish I was someone else.",
        "text": "I wish I was prettier. I wish I didn't feel like a burden, I wish I wasn't so broken. I wish I was more charismatic and not weird around strangers and I wish I wasn't so nice. I wish I didn't feel like I'm a loser with a pathetic, boring life. I wish I wasn't so hard to make friends with and I just wish I was different.",
        "label": "not suicide"
    },
]


def create_few_shot_prompt(examples, new_text, title=None):
    """Create a few-shot prompt with examples and the new text to classify."""
    prompt = "You are an expert in detecting signs of suicide ideation in text. Classify each text as either 'suicide' if the post indicates suicidal ideation or 'not suicide' if the post does not indicate suicidal tendencies, even if it expresses emotional distress. Do not explain your reasoning..\n\n"

    # Add examples
    for i, example in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        if example.get("title"):
            prompt += f"Title: {example['title']}\n"
        prompt += f"Text: {example['text']}\n"
        prompt += f"Classification: {example['label']}\n\n"

    # Add new text to classify
    prompt += "Please classify the following text:\n"
    if title:
        prompt += f"Title: {title}\n"
    prompt += f"Text: {new_text}\n"
    prompt += "Classification:"

    return prompt


def call_ollama_api(prompt, model, url="http://localhost:11434"):
    """Call Ollama API with a prompt."""
    api_endpoint = f"{url}/api/generate"

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.1  # Low temperature for more deterministic outputs
    }

    try:
        response = requests.post(api_endpoint, json=data)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None


def extract_classification(response):
    """Extract classification label from the model response."""
    response = response.lower()

    # Try various extraction methods
    if "suicide" in response and "not suicide" not in response:
        return "suicide", None
    elif "not suicide" in response:
        return "not suicide", None

    # Try to find the exact classification format
    match = re.search(
        r'classification:?\s*(suicide|not suicide)', response, re.IGNORECASE)
    if match:
        return match.group(1).lower(), None

    # Try to extract confidence scores if present
    match = re.search(r'suicide.*?(\d+\.?\d*%?)', response, re.IGNORECASE)
    suicide_score = float(match.group(1).replace('%', '')
                          ) / 100 if match else None

    match = re.search(r'not suicide.*?(\d+\.?\d*%?)', response, re.IGNORECASE)
    not_suicide_score = float(match.group(
        1).replace('%', '')) / 100 if match else None

    if suicide_score and not_suicide_score:
        total = suicide_score + not_suicide_score
        suicide_score = suicide_score / total
        return "suicide" if suicide_score > 0.5 else "not suicide", suicide_score

    # If all else fails, look for the most decisive mention
    suicide_pos = response.find("suicide")
    not_suicide_pos = response.find("not suicide")

    if suicide_pos >= 0 and not_suicide_pos >= 0:
        return "not suicide" if not_suicide_pos < suicide_pos else "suicide", None

    # Default if can't determine
    return "not suicide", None


def run_few_shot_classification(df, model_name, ollama_url="http://localhost:11434", examples=None):
    """
    Run few-shot classification using an Ollama model.

    Args:
        df (pd.DataFrame): DataFrame with text data
        model_name (str): Name of the model in Ollama
        ollama_url (str): URL for Ollama API
        examples (list): Few-shot examples, if None use default examples

    Returns:
        dict: Dictionary with results
    """
    print(f"\n{'='*50}")
    print(f"Running few-shot classification with Ollama model {model_name}")
    print(f"{'='*50}")

    if examples is None:
        examples = FEW_SHOT_EXAMPLES

    # Prepare input text
    print("Preparing input text...")
    df["input_text"] = df["title"].fillna("").astype(
        str).str.strip() + " " + df["text"].fillna("").astype(str).str.strip()

    # Map labels to binary if not already
    if 'is_suicide' in df.columns:
        if df['is_suicide'].dtype == 'object':
            df["actual_binary"] = df["is_suicide"].map({"yes": 1, "no": 0})
        else:
            df["actual_binary"] = df["is_suicide"]

    # Process each text with few-shot learning
    print(
        f"Running inference for {len(df)} samples with Ollama model {model_name}...")
    predictions = []
    scores = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        prompt = create_few_shot_prompt(
            examples, row["text"], row.get("title"))
        response = call_ollama_api(prompt, model_name, ollama_url)

        if response:
            label, score = extract_classification(response)
            pred = 1 if label == "suicide" else 0
            score = score if score is not None else (0.9 if pred == 1 else 0.1)
        else:
            pred = 0
            score = 0.1

        predictions.append(pred)
        scores.append(score)

    # Store predictions
    df[f"few_shot_{model_name}_score"] = scores
    df[f"few_shot_{model_name}_pred"] = predictions

    # Evaluate if we have ground truth
    metrics = {}
    if "actual_binary" in df.columns:
        y_true = df["actual_binary"].values
        y_pred = predictions
        y_score = scores

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

        print(f"\n=== Few-Shot Classification Results for {model_name} ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    return {
        "model_name": f"few-shot-{model_name}",
        "df": df,
        "metrics": metrics
    }
