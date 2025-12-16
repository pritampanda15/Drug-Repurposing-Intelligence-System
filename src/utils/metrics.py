"""
Evaluation metrics for link prediction.
"""

import torch
import numpy as np
from typing import Dict, List
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_mrr(y_pred_pos: torch.Tensor, y_pred_neg: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank.

    Parameters
    ----------
    y_pred_pos : torch.Tensor
        Scores for positive edges
    y_pred_neg : torch.Tensor
        Scores for negative edges

    Returns
    -------
    float
        Mean reciprocal rank
    """
    y_pred_pos = y_pred_pos.view(-1, 1)
    # Optimistic ranking: assume positive edge is ranked highest among ties
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1) + 1
    return (1.0 / optimistic_rank.float()).mean().item()


def compute_hits_at_k(
    y_pred_pos: torch.Tensor,
    y_pred_neg: torch.Tensor,
    k: int
) -> float:
    """
    Compute Hits@K metric.

    Parameters
    ----------
    y_pred_pos : torch.Tensor
        Scores for positive edges
    y_pred_neg : torch.Tensor
        Scores for negative edges
    k : int
        Cutoff for hits

    Returns
    -------
    float
        Proportion of positive edges in top-k
    """
    y_pred_pos = y_pred_pos.view(-1, 1)
    # Optimistic ranking
    rank = (y_pred_neg >= y_pred_pos).sum(dim=1) + 1
    return (rank <= k).float().mean().item()


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute ROC-AUC score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted scores

    Returns
    -------
    float
        ROC-AUC score
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.0


def compute_metrics(
    y_pred_pos: torch.Tensor,
    y_pred_neg: torch.Tensor,
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_pred_pos : torch.Tensor
        Scores for positive edges
    y_pred_neg : torch.Tensor
        Scores for negative edges
    y_true : np.ndarray, optional
        Ground truth labels for AUC
    y_pred : np.ndarray, optional
        Predicted scores for AUC

    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and values
    """
    metrics = {
        'mrr': compute_mrr(y_pred_pos, y_pred_neg),
        'hits@1': compute_hits_at_k(y_pred_pos, y_pred_neg, 1),
        'hits@5': compute_hits_at_k(y_pred_pos, y_pred_neg, 5),
        'hits@10': compute_hits_at_k(y_pred_pos, y_pred_neg, 10),
    }

    if y_true is not None and y_pred is not None:
        metrics['auc'] = compute_auc(y_true, y_pred)

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Pretty print metrics.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metrics
    prefix : str
        Prefix for output (e.g., "Train", "Val", "Test")
    """
    print(f"\n{prefix} Metrics:" if prefix else "\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
