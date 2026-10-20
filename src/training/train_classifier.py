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


def train_one_config(synthetic_ratio, config, transform, device, run):
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
    train_dataset = ImageFolderBinary(train_pairs, transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

    test_positive = list_images("data/test_real/positive", label=1)
    test_negative = list_images("data/test_real/negative", label=0)
    test_dataset = ImageFolderBinary(test_positive + test_negative, transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_results = []
    for ratio in config["classifier"]["synthetic_ratios"]:
        print(f"\n=== Training classifier, synthetic_ratio = {ratio} ===")
        set_seed(config["project"]["seed"])  # reset seed so each ratio starts identically
        metrics = train_one_config(ratio, config, transform, device, run)
        print(metrics)
        all_results.append(metrics)

    columns = ["synthetic_ratio", "f1", "precision", "recall", "pr_auc",
               "n_real_positive_train", "n_synthetic_positive_train"]
    rows = [[m[c] for c in columns] for m in all_results]
    log_results_table(run, columns, rows, "classifier_results_by_ratio")

    baseline = next(m for m in all_results if m["synthetic_ratio"] == 0.0)
    best = max(all_results, key=lambda m: m["f1"])
    print(f"\nBaseline (real-only) F1: {baseline['f1']:.4f}")
    print(f"Best F1: {best['f1']:.4f} at synthetic_ratio={best['synthetic_ratio']}")
    run.log({
        "summary/baseline_f1": baseline["f1"],
        "summary/best_f1": best["f1"],
        "summary/best_ratio": best["synthetic_ratio"],
        "summary/f1_improvement": best["f1"] - baseline["f1"],
    })

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)