"""
Shared evaluation utilities for the downstream classifier. Imported by
train_classifier.py — kept separate so evaluation logic is identical
regardless of which synthetic ratio produced the model being tested.
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score, recall_score, precision_score


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

    return {
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "pr_auc": float(average_precision_score(all_labels, all_probs)),
        "n_positive": int(all_labels.sum()),
        "n_total": int(len(all_labels)),
    }