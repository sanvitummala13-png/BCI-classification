"""
Evaluation module for Alzheimer's EEG Detection system.

This module provides the Evaluator class and related utilities for computing
performance metrics, generating evaluation reports, and comparing models.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.6
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResults:
    """
    Container for evaluation results from a single model evaluation.
    
    Attributes:
        accuracy: Overall classification accuracy
        precision: Precision score (TP / (TP + FP))
        recall: Recall/Sensitivity score (TP / (TP + FN))
        f1_score: F1 score (harmonic mean of precision and recall)
        roc_auc: Area under ROC curve
        specificity: Specificity score (TN / (TN + FP))
        confusion_matrix: 2x2 confusion matrix
        per_class_metrics: Metrics computed separately for each class
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    specificity: float
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'specificity': self.specificity,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'per_class_metrics': self.per_class_metrics
        }


@dataclass
class CrossValidationResults:
    """
    Container for cross-validation evaluation results.
    
    Attributes:
        fold_results: List of EvaluationResults for each fold
        mean_metrics: Mean of metrics across folds
        std_metrics: Standard deviation of metrics across folds
        best_fold: Index of the best performing fold
    """
    fold_results: List[EvaluationResults]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_fold: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'fold_results': [r.to_dict() for r in self.fold_results],
            'mean_metrics': self.mean_metrics,
            'std_metrics': self.std_metrics,
            'best_fold': self.best_fold
        }



def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        
    Returns:
        2x2 confusion matrix where:
        - [0,0] = True Negatives (TN)
        - [0,1] = False Positives (FP)
        - [1,0] = False Negatives (FN)
        - [1,1] = True Positives (TP)
        
    Requirements: 5.3
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Compute confusion matrix elements
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score in range [0, 1]
        
    Requirements: 5.1
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) == 0:
        return 0.0
    
    return np.mean(y_true == y_pred)


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision score.
    
    Precision = TP / (TP + FP)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Precision score in range [0, 1]
        
    Requirements: 5.1
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall (sensitivity) score.
    
    Recall = TP / (TP + FN)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Recall score in range [0, 1]
        
    Requirements: 5.1
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute F1 score.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        F1 score in range [0, 1]
        
    Requirements: 5.1
    """
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute specificity score.
    
    Specificity = TN / (TN + FP)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Specificity score in range [0, 1]
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    
    if tn + fp == 0:
        return 0.0
    
    return tn / (tn + fp)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each class.
    
    For each class, treats that class as the positive class and computes
    precision, recall, F1, and support.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics for each class:
        {
            'Control': {'precision': ..., 'recall': ..., 'f1': ..., 'support': ...},
            'AD': {'precision': ..., 'recall': ..., 'f1': ..., 'support': ...}
        }
        
    Requirements: 5.5
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    per_class = {}
    class_names = {0: 'Control', 1: 'AD'}
    
    for class_label, class_name in class_names.items():
        # Treat current class as positive
        y_true_binary = (y_true == class_label).astype(int)
        y_pred_binary = (y_pred == class_label).astype(int)
        
        # Compute metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(np.sum(y_true == class_label))
        
        per_class[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    return per_class


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class (optional, for ROC-AUC)
        
    Returns:
        Dictionary containing all metrics
        
    Requirements: 5.1, 5.3
    """
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred),
        'recall': compute_recall(y_true, y_pred),
        'f1_score': compute_f1_score(y_true, y_pred),
        'specificity': compute_specificity(y_true, y_pred)
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_prob is not None:
        metrics['roc_auc'] = compute_roc_auc(y_true, y_prob)
    
    return metrics



def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (Receiver Operating Characteristic).
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_prob: Predicted probabilities for positive class (class 1)
        
    Returns:
        Tuple of (fpr, tpr, thresholds):
        - fpr: False positive rates at each threshold
        - tpr: True positive rates at each threshold
        - thresholds: Threshold values used
        
    Requirements: 5.2
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    
    # Count positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        # Edge case: only one class present
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    
    # Sort by probability in descending order
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]
    
    # Compute TPR and FPR incrementally
    tpr_list = [0.0]
    fpr_list = [0.0]
    threshold_list = [y_prob_sorted[0] + 1e-10]  # Start above highest prob
    
    tp_count = 0
    fp_count = 0
    
    for i, (label, prob) in enumerate(zip(y_true_sorted, y_prob_sorted)):
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        
        tpr = tp_count / n_pos
        fpr = fp_count / n_neg
        
        # Only add point if it's different from the last one
        if tpr != tpr_list[-1] or fpr != fpr_list[-1]:
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            threshold_list.append(prob)
    
    # Ensure curve ends at (1, 1)
    if fpr_list[-1] != 1.0 or tpr_list[-1] != 1.0:
        fpr_list.append(1.0)
        tpr_list.append(1.0)
        threshold_list.append(0.0)
    
    return np.array(fpr_list), np.array(tpr_list), np.array(threshold_list)


def compute_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> float:
    """
    Compute Area Under ROC Curve (AUC).
    
    Uses the trapezoidal rule to compute the area under the ROC curve.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_prob: Predicted probabilities for positive class (class 1)
        
    Returns:
        ROC-AUC score in range [0, 1], where 0.5 represents random guessing
        
    Requirements: 5.2
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    
    # Handle edge cases
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        logger.warning("Only one class present in y_true. ROC-AUC is undefined.")
        return 0.5
    
    # Compute ROC curve
    fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
    
    # Compute AUC using trapezoidal rule
    # Sort by FPR to ensure proper integration
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Trapezoidal integration
    auc = np.trapezoid(tpr_sorted, fpr_sorted)
    
    # Ensure AUC is in valid range
    return float(np.clip(auc, 0.0, 1.0))



def compute_cross_validation_statistics(
    fold_results: List[EvaluationResults]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute mean and standard deviation of metrics across cross-validation folds.
    
    Args:
        fold_results: List of EvaluationResults from each fold
        
    Returns:
        Tuple of (mean_metrics, std_metrics) dictionaries
        
    Requirements: 5.4
    """
    if not fold_results:
        return {}, {}
    
    # Extract metrics from each fold
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'specificity']
    
    metrics_by_name = {name: [] for name in metric_names}
    
    for result in fold_results:
        metrics_by_name['accuracy'].append(result.accuracy)
        metrics_by_name['precision'].append(result.precision)
        metrics_by_name['recall'].append(result.recall)
        metrics_by_name['f1_score'].append(result.f1_score)
        metrics_by_name['roc_auc'].append(result.roc_auc)
        metrics_by_name['specificity'].append(result.specificity)
    
    # Compute mean and std for each metric
    mean_metrics = {}
    std_metrics = {}
    
    for name, values in metrics_by_name.items():
        values_array = np.array(values)
        mean_metrics[name] = float(np.mean(values_array))
        # Use sample standard deviation (ddof=1) for unbiased estimate
        std_metrics[name] = float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0
    
    return mean_metrics, std_metrics


def generate_cross_validation_summary(
    fold_results: List[EvaluationResults]
) -> CrossValidationResults:
    """
    Generate a comprehensive cross-validation summary report.
    
    Args:
        fold_results: List of EvaluationResults from each fold
        
    Returns:
        CrossValidationResults containing aggregated statistics
        
    Requirements: 5.4
    """
    mean_metrics, std_metrics = compute_cross_validation_statistics(fold_results)
    
    # Find best fold based on F1 score
    best_fold = 0
    best_f1 = -1.0
    for i, result in enumerate(fold_results):
        if result.f1_score > best_f1:
            best_f1 = result.f1_score
            best_fold = i
    
    return CrossValidationResults(
        fold_results=fold_results,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        best_fold=best_fold
    )



def compare_models(
    model_results: Dict[str, EvaluationResults]
) -> Dict[str, Any]:
    """
    Compare multiple models using identical metrics.
    
    Args:
        model_results: Dictionary mapping model names to their EvaluationResults
        
    Returns:
        Dictionary containing:
        - 'comparison_table': List of dicts with model metrics
        - 'best_model': Name of best performing model (by F1 score)
        - 'rankings': Dict mapping metric names to ranked model lists
        
    Requirements: 6.6
    """
    if not model_results:
        return {'comparison_table': [], 'best_model': None, 'rankings': {}}
    
    # Build comparison table
    comparison_table = []
    for model_name, results in model_results.items():
        row = {
            'model': model_name,
            'accuracy': results.accuracy,
            'precision': results.precision,
            'recall': results.recall,
            'f1_score': results.f1_score,
            'roc_auc': results.roc_auc,
            'specificity': results.specificity
        }
        comparison_table.append(row)
    
    # Sort by F1 score (descending)
    comparison_table.sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Find best model
    best_model = comparison_table[0]['model'] if comparison_table else None
    
    # Generate rankings for each metric
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'specificity']
    rankings = {}
    
    for metric in metric_names:
        sorted_models = sorted(
            model_results.keys(),
            key=lambda m: getattr(model_results[m], metric),
            reverse=True
        )
        rankings[metric] = sorted_models
    
    return {
        'comparison_table': comparison_table,
        'best_model': best_model,
        'rankings': rankings
    }


def format_comparison_table(
    model_results: Dict[str, EvaluationResults],
    include_header: bool = True
) -> str:
    """
    Format model comparison as a readable table string.
    
    Args:
        model_results: Dictionary mapping model names to their EvaluationResults
        include_header: Whether to include column headers
        
    Returns:
        Formatted table string
        
    Requirements: 6.6
    """
    if not model_results:
        return "No models to compare."
    
    # Column widths
    col_widths = {
        'model': max(15, max(len(name) for name in model_results.keys()) + 2),
        'accuracy': 10,
        'precision': 10,
        'recall': 10,
        'f1_score': 10,
        'roc_auc': 10,
        'specificity': 12
    }
    
    lines = []
    
    # Header
    if include_header:
        header = (
            f"{'Model':<{col_widths['model']}}"
            f"{'Accuracy':>{col_widths['accuracy']}}"
            f"{'Precision':>{col_widths['precision']}}"
            f"{'Recall':>{col_widths['recall']}}"
            f"{'F1':>{col_widths['f1_score']}}"
            f"{'ROC-AUC':>{col_widths['roc_auc']}}"
            f"{'Specificity':>{col_widths['specificity']}}"
        )
        lines.append(header)
        lines.append('-' * len(header))
    
    # Sort by F1 score
    sorted_models = sorted(
        model_results.items(),
        key=lambda x: x[1].f1_score,
        reverse=True
    )
    
    # Data rows
    for model_name, results in sorted_models:
        row = (
            f"{model_name:<{col_widths['model']}}"
            f"{results.accuracy:>{col_widths['accuracy']}.4f}"
            f"{results.precision:>{col_widths['precision']}.4f}"
            f"{results.recall:>{col_widths['recall']}.4f}"
            f"{results.f1_score:>{col_widths['f1_score']}.4f}"
            f"{results.roc_auc:>{col_widths['roc_auc']}.4f}"
            f"{results.specificity:>{col_widths['specificity']}.4f}"
        )
        lines.append(row)
    
    return '\n'.join(lines)



def save_results_json(
    results: Union[EvaluationResults, CrossValidationResults, Dict[str, Any]],
    filepath: str
) -> None:
    """
    Save evaluation results to JSON format.
    
    Args:
        results: Evaluation results to save
        filepath: Path to save the JSON file
        
    Requirements: 5.6
    """
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict if needed
    if hasattr(results, 'to_dict'):
        data = results.to_dict()
    else:
        data = results
    
    # Add metadata
    data['saved_at'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")


def save_results_csv(
    results: Union[EvaluationResults, Dict[str, EvaluationResults]],
    filepath: str
) -> None:
    """
    Save evaluation results to CSV format.
    
    Args:
        results: Single EvaluationResults or dict of model results
        filepath: Path to save the CSV file
        
    Requirements: 5.6
    """
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Handle single result vs multiple models
    if isinstance(results, EvaluationResults):
        model_results = {'model': results}
    else:
        model_results = results
    
    # Write CSV
    with open(filepath, 'w') as f:
        # Header
        f.write('model,accuracy,precision,recall,f1_score,roc_auc,specificity\n')
        
        # Data rows
        for model_name, result in model_results.items():
            f.write(
                f"{model_name},"
                f"{result.accuracy:.6f},"
                f"{result.precision:.6f},"
                f"{result.recall:.6f},"
                f"{result.f1_score:.6f},"
                f"{result.roc_auc:.6f},"
                f"{result.specificity:.6f}\n"
            )
    
    logger.info(f"Results saved to {filepath}")


def save_confusion_matrix_image(
    confusion_matrix: np.ndarray,
    filepath: str,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Save confusion matrix as an image.
    
    Args:
        confusion_matrix: 2x2 confusion matrix
        filepath: Path to save the image
        class_names: Names for classes (default: ['Control', 'AD'])
        title: Title for the plot
        figsize: Figure size in inches
        
    Requirements: 5.6
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Skipping confusion matrix image.")
        return
    
    if class_names is None:
        class_names = ['Control', 'AD']
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(confusion_matrix[i, j], 'd'),
                ha='center', va='center',
                color='white' if confusion_matrix[i, j] > thresh else 'black'
            )
    
    fig.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Confusion matrix saved to {filepath}")


def save_roc_curve_image(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    filepath: str,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Save ROC curve as an image.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities for positive class
        filepath: Path to save the image
        title: Title for the plot
        figsize: Figure size in inches
        
    Requirements: 5.6
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Skipping ROC curve image.")
        return
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Compute ROC curve
    fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
    auc = compute_roc_auc(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"ROC curve saved to {filepath}")



class Evaluator:
    """
    Complete evaluation pipeline for EEG classification models.
    
    Orchestrates model evaluation with support for:
    - Computing comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
    - Generating confusion matrices
    - Cross-validation statistics
    - Model comparison
    - Saving results in multiple formats
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.6
    """
    
    def __init__(
        self,
        output_dir: str = 'results',
        save_visualizations: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory for saving results and visualizations
            save_visualizations: Whether to save confusion matrices and ROC curves
        """
        self.output_dir = output_dir
        self.save_visualizations = save_visualizations
        self.model_results: Dict[str, EvaluationResults] = {}
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Evaluator initialized with output directory: {output_dir}")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> EvaluationResults:
        """
        Evaluate model predictions and compute all metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities for positive class (optional)
            model_name: Name identifier for the model
            
        Returns:
            EvaluationResults containing all computed metrics
            
        Requirements: 5.1, 5.2, 5.3, 5.5
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Compute all metrics
        accuracy = compute_accuracy(y_true, y_pred)
        precision = compute_precision(y_true, y_pred)
        recall = compute_recall(y_true, y_pred)
        f1 = compute_f1_score(y_true, y_pred)
        specificity = compute_specificity(y_true, y_pred)
        confusion_mat = compute_confusion_matrix(y_true, y_pred)
        per_class = compute_per_class_metrics(y_true, y_pred)
        
        # Compute ROC-AUC if probabilities provided
        if y_prob is not None:
            y_prob = np.asarray(y_prob).flatten()
            roc_auc = compute_roc_auc(y_true, y_prob)
        else:
            roc_auc = 0.0
            logger.warning(
                f"No probabilities provided for {model_name}. ROC-AUC set to 0.0"
            )
        
        results = EvaluationResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            specificity=specificity,
            confusion_matrix=confusion_mat,
            per_class_metrics=per_class
        )
        
        # Store results
        self.model_results[model_name] = results
        
        # Log results
        logger.info(
            f"Evaluation results for {model_name}: "
            f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
            f"Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}"
        )
        
        # Save visualizations if enabled
        if self.save_visualizations:
            self._save_model_visualizations(
                model_name, confusion_mat, y_true, y_prob
            )
        
        return results
    
    def _save_model_visualizations(
        self,
        model_name: str,
        confusion_mat: np.ndarray,
        y_true: np.ndarray,
        y_prob: Optional[np.ndarray]
    ) -> None:
        """Save confusion matrix and ROC curve for a model."""
        model_dir = os.path.join(self.output_dir, model_name)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix
        cm_path = os.path.join(model_dir, 'confusion_matrix.png')
        save_confusion_matrix_image(
            confusion_mat, cm_path,
            title=f"Confusion Matrix - {model_name}"
        )
        
        # Save ROC curve if probabilities available
        if y_prob is not None:
            roc_path = os.path.join(model_dir, 'roc_curve.png')
            save_roc_curve_image(
                y_true, y_prob, roc_path,
                title=f"ROC Curve - {model_name}"
            )
    
    def evaluate_cross_validation(
        self,
        fold_predictions: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        model_name: str = "model"
    ) -> CrossValidationResults:
        """
        Evaluate cross-validation results across multiple folds.
        
        Args:
            fold_predictions: List of (y_true, y_pred, y_prob) tuples for each fold
            model_name: Name identifier for the model
            
        Returns:
            CrossValidationResults with aggregated statistics
            
        Requirements: 5.4
        """
        fold_results = []
        
        for fold_idx, (y_true, y_pred, y_prob) in enumerate(fold_predictions):
            fold_name = f"{model_name}_fold_{fold_idx + 1}"
            result = self.evaluate(y_true, y_pred, y_prob, fold_name)
            fold_results.append(result)
        
        # Generate summary
        cv_results = generate_cross_validation_summary(fold_results)
        
        # Log summary
        logger.info(
            f"Cross-validation summary for {model_name}: "
            f"Mean F1={cv_results.mean_metrics.get('f1_score', 0):.4f} "
            f"± {cv_results.std_metrics.get('f1_score', 0):.4f}"
        )
        
        return cv_results
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare all evaluated models.
        
        Returns:
            Comparison results including rankings and best model
            
        Requirements: 6.6
        """
        if not self.model_results:
            logger.warning("No models to compare. Run evaluate() first.")
            return {'comparison_table': [], 'best_model': None, 'rankings': {}}
        
        comparison = compare_models(self.model_results)
        
        # Log comparison
        logger.info(f"Model comparison - Best model: {comparison['best_model']}")
        logger.info("\n" + format_comparison_table(self.model_results))
        
        return comparison
    
    def save_results(
        self,
        filename_prefix: str = "evaluation_results"
    ) -> Dict[str, str]:
        """
        Save all evaluation results to files.
        
        Args:
            filename_prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping result types to file paths
            
        Requirements: 5.6
        """
        saved_files = {}
        
        # Save individual model results as JSON
        for model_name, results in self.model_results.items():
            json_path = os.path.join(
                self.output_dir, model_name, f"{filename_prefix}.json"
            )
            save_results_json(results, json_path)
            saved_files[f"{model_name}_json"] = json_path
        
        # Save comparison CSV if multiple models
        if len(self.model_results) > 1:
            csv_path = os.path.join(
                self.output_dir, f"{filename_prefix}_comparison.csv"
            )
            save_results_csv(self.model_results, csv_path)
            saved_files['comparison_csv'] = csv_path
            
            # Save comparison JSON
            comparison = compare_models(self.model_results)
            comparison_json_path = os.path.join(
                self.output_dir, f"{filename_prefix}_comparison.json"
            )
            save_results_json(comparison, comparison_json_path)
            saved_files['comparison_json'] = comparison_json_path
        
        logger.info(f"Results saved to {len(saved_files)} files")
        return saved_files
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of all evaluations.
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Individual model results
        for model_name, results in self.model_results.items():
            lines.append("-" * 40)
            lines.append(f"Model: {model_name}")
            lines.append("-" * 40)
            lines.append(f"  Accuracy:    {results.accuracy:.4f}")
            lines.append(f"  Precision:   {results.precision:.4f}")
            lines.append(f"  Recall:      {results.recall:.4f}")
            lines.append(f"  F1 Score:    {results.f1_score:.4f}")
            lines.append(f"  ROC-AUC:     {results.roc_auc:.4f}")
            lines.append(f"  Specificity: {results.specificity:.4f}")
            lines.append("")
            lines.append("  Confusion Matrix:")
            lines.append(f"    TN={results.confusion_matrix[0,0]:4d}  FP={results.confusion_matrix[0,1]:4d}")
            lines.append(f"    FN={results.confusion_matrix[1,0]:4d}  TP={results.confusion_matrix[1,1]:4d}")
            lines.append("")
            lines.append("  Per-Class Metrics:")
            for class_name, metrics in results.per_class_metrics.items():
                lines.append(
                    f"    {class_name}: P={metrics['precision']:.4f}, "
                    f"R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
                    f"Support={metrics['support']}"
                )
            lines.append("")
        
        # Model comparison
        if len(self.model_results) > 1:
            lines.append("=" * 60)
            lines.append("MODEL COMPARISON")
            lines.append("=" * 60)
            lines.append("")
            lines.append(format_comparison_table(self.model_results))
            lines.append("")
            
            comparison = compare_models(self.model_results)
            lines.append(f"Best Model (by F1): {comparison['best_model']}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    def get_results(self, model_name: str) -> Optional[EvaluationResults]:
        """
        Get evaluation results for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            EvaluationResults or None if model not found
        """
        return self.model_results.get(model_name)
    
    def get_all_results(self) -> Dict[str, EvaluationResults]:
        """
        Get all evaluation results.
        
        Returns:
            Dictionary mapping model names to their results
        """
        return self.model_results.copy()
    
    def clear_results(self) -> None:
        """Clear all stored evaluation results."""
        self.model_results.clear()
        logger.info("All evaluation results cleared")

