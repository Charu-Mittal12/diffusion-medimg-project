"""
Shared evaluation utilities for the downstream classifier. Imported by
train_classifier.py — kept separate so evaluation logic is identical
regardless of which synthetic ratio produced the model being tested.

Reports metrics at two operating points:
  - threshold=0.5 (the standard default, what you'd ship without any
    tuning)
  - best_threshold (the threshold on THIS test set's own precision-recall
    curve that maximizes F1) — this is an upper bound on what the model
    could achieve with ideal threshold calibration, not an unbiased
    estimate (a real deployment would pick the threshold on a separate
    validation set, not the test set). We report both so the gap between
    them tells you whether a ratio's problem is the model's ranking
    (both low) or just an uncalibrated default threshold (0.5 low,
    best_threshold much higher).
"""
import numpy as np
import torch
from sklearn.metrics import (
    f1_score, average_precision_score, recall_score, precision_score,
    precision_recall_curve,
)


def _best_f1_threshold(labels: np.ndarray, probs: np.ndarray):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve returns one more point than thresholds (it
    # appends the (precision=1, recall=0) point with no threshold) — drop it.
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls + 1e-12),
        0.0,
    )
    if len(f1s) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


@torch.no_grad()
def evaluate(model, dataloader, device) -> dict:
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probability of positive (pathology) class
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    best_threshold, best_f1 = _best_f1_threshold(all_labels, all_probs)
    best_preds = (all_probs >= best_threshold).astype(int)

    return {
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "pr_auc": float(average_precision_score(all_labels, all_probs)),
        "best_threshold": best_threshold,
        "best_threshold_f1": best_f1,
        "best_threshold_precision": float(precision_score(all_labels, best_preds, zero_division=0)),
        "best_threshold_recall": float(recall_score(all_labels, best_preds, zero_division=0)),
        "n_positive": int(all_labels.sum()),
        "n_total": int(len(all_labels)),
    }
