"""
Visualization utilities for model evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, labels=['no', 'yes'], title="Confusion Matrix", save_path=None):
    """Plot a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.colormaps["Blues"])
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba, title="ROC Curve", label=None, save_path=None, ax=None):
    """
    Plot a ROC curve with AUC score.
    If ax is provided, plots on the existing axes.
    """
    from sklearn.metrics import roc_auc_score

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

    label = label or f'ROC curve (AUC = {auc:.4f})'
    ax.plot(fpr, tpr, lw=2, label=label)

    if ax == plt.gca():  # Only add these elements if we created the plot
        ax.plot([0, 1], [0, 1], color='gray',
                linestyle='--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    return ax, auc


def plot_all_roc_curves(results_list, title="ROC Curves Comparison", save_path=None):
    """Plot all ROC curves on a single plot."""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Plot the random baseline once
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random')

    for result in sorted(results_list, key=lambda x: x.get('metrics', {}).get('auc', 0), reverse=True):
        metrics = result.get('metrics', {})
        if not metrics:
            continue

        model_name = result.get('model_name', 'Unknown')
        auc = metrics.get('auc', 0)

        # Handle different key naming conventions between traditional ML and LLMs
        y_true = None
        y_score = None

        # For traditional ML results (y_test is stored and y_proba is the score)
        if 'y_test' in metrics and 'y_proba' in metrics:
            y_true = metrics['y_test']
            y_score = metrics['y_proba']
        # Alternative naming in traditional ML
        elif 'y_true' in metrics and 'y_proba' in metrics:
            y_true = metrics['y_true']
            y_score = metrics['y_proba']
        # For LLM results (y_true and y_score naming)
        elif 'y_true' in metrics and 'y_score' in metrics:
            y_true = metrics['y_true']
            y_score = metrics['y_score']

        if y_true is not None and y_score is not None:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_list, metric='auc', title=None, save_path=None):
    """Plot a comparison of models based on a metric."""
    # Extract data for plotting
    model_names = []
    metric_values = []

    for result in results_list:
        metrics = result.get('metrics', {})
        if not metrics:
            continue

        model_name = result.get('model_name', 'Unknown')
        metric_value = metrics.get(metric, 0)

        model_names.append(model_name)
        metric_values.append(metric_value)

    # Sort by metric value
    sorted_indices = np.argsort(metric_values)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    metric_values = [metric_values[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))

    # Set a meaningful title if none provided
    if title is None:
        title = f'Model Comparison ({metric.upper()})'

    bars = plt.bar(model_names, metric_values,
                   color=plt.cm.tab10.colors[:len(model_names)])
    plt.title(title)
    plt.ylabel(metric.upper())
    plt.xlabel('Model')
    plt.ylim(0.5 if metric.lower() == 'auc' else 0.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def display_best_model_metrics(results_list):
    """Display detailed metrics for the best model."""
    # Find best model based on AUC
    best_result = max(results_list, key=lambda x: x.get(
        'metrics', {}).get('auc', 0))
    metrics = best_result.get('metrics', {})
    model_name = best_result.get('model_name', 'Unknown')

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"AUC:       {metrics.get('auc', 0):.4f}")
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
    print("\nConfusion Matrix:")
    print(metrics.get('confusion_matrix', 'Not available'))

    # Plot confusion matrix for best model
    if 'y_true' in metrics and 'y_pred' in metrics:
        plot_confusion_matrix(
            metrics['y_true'],
            metrics['y_pred'],
            title=f"Confusion Matrix - Best Model: {model_name}",
            save_path=f'best_model_confusion_matrix.png'
        )

    return best_result
