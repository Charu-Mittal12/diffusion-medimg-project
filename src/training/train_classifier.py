"""
Stage 5 — Downstream classifier experiment: the actual test of whether
synthetic data helps. Trains a separate ResNet-18 at each configured
synthetic_ratio (0.0 = real-data-only baseline), evaluating every one
on the SAME held-out real test set so results are directly comparable.
"""
import argparse
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from src.utils.seed import set_seed, load_config
from src.utils.wandb_logger import init_run, log_results_table
from src.evaluation.classify_eval import evaluate


class ImageFolderBinary(Dataset):
    """label=1 for images in `positive_dir`, label=0 for images in `negative_dir`.
    If only one dir is given, all images get the fixed `label`."""

    def __init__(self, paths_labels, transform):
        self.paths_labels = paths_labels  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        path, label = self.paths_labels[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def list_images(directory, label):
    return [(os.path.join(directory, f), label)
            for f in os.listdir(directory) if f.endswith(".png")]


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # binary head
    return model.to(device)


def train_one_config(synthetic_ratio, config, train_transform, test_transform, device, run):
    cfg = config["classifier"]
    data_cfg = config["data"]

    real_positive = list_images(data_cfg["image_export_dir"], label=1)
    real_negative_pool = list_images("data/classifier_negatives", label=0)

    n_synthetic = int(len(real_positive) * synthetic_ratio)
    synthetic_positive = []
    if n_synthetic > 0:
        synth_dir = config["generation"]["output_dir"]
        if not os.path.isdir(synth_dir) or not any(f.endswith(".png") for f in os.listdir(synth_dir)):
            print(f"  Skipping synthetic_ratio={synthetic_ratio}: no synthetic images found yet "
                  f"in {synth_dir} (run generate_samples.py first, then rerun this script).")
            return None
        synth_files = sorted(f for f in os.listdir(synth_dir) if f.endswith(".png"))
        random.shuffle(synth_files)
        synthetic_positive = [
            (os.path.join(synth_dir, f), 1) for f in synth_files[:n_synthetic]
        ]

    train_pairs = real_positive + synthetic_positive + real_negative_pool
    train_dataset = ImageFolderBinary(train_pairs, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

    test_positive = list_images("data/test_real/positive", label=1)
    test_negative = list_images("data/test_real/negative", label=0)
    test_dataset = ImageFolderBinary(test_positive + test_negative, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # Class-weighted loss: with only ~17% positives, unweighted CE barely
    # penalizes the model for ignoring the minority class, so it defaults
    # to predicting negative almost everywhere (exactly what we saw:
    # precision pinned at 1.0, recall stuck ~0.16 across several ratios).
    # Inverse-frequency weighting (sklearn's "balanced" scheme) pushes
    # back against that.
    n_pos_train = len(real_positive) + len(synthetic_positive)
    n_neg_train = len(real_negative_pool)
    n_total_train = n_pos_train + n_neg_train
    class_weights = torch.tensor([
        n_total_train / (2.0 * n_neg_train),  # weight for class 0 (negative)
        n_total_train / (2.0 * n_pos_train),  # weight for class 1 (positive)
    ], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if run is not None:
            run.log({
                f"ratio_{synthetic_ratio}/epoch": epoch,
                f"ratio_{synthetic_ratio}/train_loss": epoch_loss / len(train_loader),
            })

    metrics = evaluate(model, test_loader, device)
    metrics["synthetic_ratio"] = synthetic_ratio
    metrics["n_real_positive_train"] = len(real_positive)
    metrics["n_synthetic_positive_train"] = n_synthetic
    return metrics


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    run = init_run(config, job_type="classifier-training", run_name="train_classifier_all_ratios")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mild augmentation on the training set only (flips/small rotation are
    # anatomically reasonable for a chest X-ray). The test set stays
    # untouched so evaluation is always on the real, unaugmented held-out
    # images.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_results = []
    skipped_ratios = []
    for ratio in config["classifier"]["synthetic_ratios"]:
        print(f"\n=== Training classifier, synthetic_ratio = {ratio} ===")
        set_seed(config["project"]["seed"])  # reset seed so each ratio starts identically
        metrics = train_one_config(ratio, config, train_transform, test_transform, device, run)
        if metrics is None:
            skipped_ratios.append(ratio)
            continue
        print(metrics)
        all_results.append(metrics)

    if not all_results:
        print("\nNo ratios produced results (synthetic images not generated yet?). Nothing to log.")
        run.finish()
        return

    columns = ["synthetic_ratio", "f1", "precision", "recall", "pr_auc",
               "best_threshold", "best_threshold_f1", "best_threshold_precision", "best_threshold_recall",
               "n_real_positive_train", "n_synthetic_positive_train"]
    rows = [[m[c] for c in columns] for m in all_results]
    log_results_table(run, columns, rows, "classifier_results_by_ratio")

    baseline = next((m for m in all_results if m["synthetic_ratio"] == 0.0), None)
    best = max(all_results, key=lambda m: m["f1"])
    best_tuned = max(all_results, key=lambda m: m["best_threshold_f1"])
    if baseline is not None:
        print(f"\nBaseline (real-only) F1 @0.5: {baseline['f1']:.4f}  |  tuned-threshold F1: {baseline['best_threshold_f1']:.4f}")
        print(f"Best F1 @0.5: {best['f1']:.4f} at synthetic_ratio={best['synthetic_ratio']}")
        print(f"Best tuned-threshold F1: {best_tuned['best_threshold_f1']:.4f} "
              f"at synthetic_ratio={best_tuned['synthetic_ratio']} (threshold={best_tuned['best_threshold']:.3f})")
        run.log({
            "summary/baseline_f1": baseline["f1"],
            "summary/baseline_best_threshold_f1": baseline["best_threshold_f1"],
            "summary/best_f1": best["f1"],
            "summary/best_ratio": best["synthetic_ratio"],
            "summary/f1_improvement": best["f1"] - baseline["f1"],
            "summary/best_tuned_f1": best_tuned["best_threshold_f1"],
            "summary/best_tuned_ratio": best_tuned["synthetic_ratio"],
        })
    else:
        print(f"\nBest F1 so far: {best['f1']:.4f} at synthetic_ratio={best['synthetic_ratio']} "
              "(no synthetic_ratio=0.0 baseline run included this time)")

    if skipped_ratios:
        print(f"\nSkipped ratios (no synthetic images yet): {skipped_ratios}. "
              "Run generate_samples.py, then rerun this script to fill them in.")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)