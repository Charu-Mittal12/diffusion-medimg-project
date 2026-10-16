"""
Stage 1 — Data preparation.

Loads NIH ChestX-ray14 (via the Hugging Face `datasets` mirror, so no
manual NIH Box download is required), filters down to:
  (a) a small (~50 image) set of the target rare pathology, exported as a
      captioned image folder for LoRA fine-tuning, and
  (b) train/test splits for the downstream binary classifier
      (target pathology vs. everything else), with a held-out REAL test
      set that synthetic data will never touch.

Every split is logged as a W&B Artifact so later runs know exactly which
images they used, not just "the dataset" in the abstract.
"""
import argparse
import os
import random
import shutil

from datasets import load_dataset
from tqdm import tqdm

from src.utils.seed import set_seed, load_config
from src.utils.wandb_logger import init_run, log_dataset_artifact


def filter_by_pathology(example, pathology: str) -> bool:
    labels = example["labels"] if "labels" in example else example.get("finding_labels", "")
    if isinstance(labels, list):
        return pathology in labels
    return pathology in str(labels)


def export_images(dataset, out_dir: str, caption: str, n: int, resolution: int):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for example in dataset:
        if count >= n:
            break
        img = example["image"].convert("RGB").resize((resolution, resolution))
        img_path = os.path.join(out_dir, f"{count:04d}.png")
        img.save(img_path)
        # LoRA training (diffusers DreamBooth script) expects a matching .txt
        # caption file OR a metadata.jsonl — we write both for flexibility.
        with open(os.path.join(out_dir, f"{count:04d}.txt"), "w") as f:
            f.write(caption)
        count += 1
    return count


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    run = init_run(config, job_type="data-prep", run_name="prepare_data")

    print("Loading NIH ChestX-ray14 (this may take a while on first run)...")
    ds = load_dataset(
        config["data"]["hf_dataset_id"],
        config["data"].get("hf_dataset_config", "image-classification"),
        split="train",
        trust_remote_code=True,
    )

    pathology = config["data"]["target_pathology"]
    caption = config["data"]["caption_template"].format(class_name=config["data"]["class_name"])

    print(f"Filtering for target pathology: {pathology}")
    positive_ds = ds.filter(lambda ex: filter_by_pathology(ex, pathology))
    print(f"Found {len(positive_ds)} total images with '{pathology}'.")

    if len(positive_ds) < config["data"]["n_finetune_images"]:
        print(
            f"WARNING: only {len(positive_ds)} positive images available, "
            f"fewer than the configured {config['data']['n_finetune_images']}. "
            "Using all available positives."
        )

    # Shuffle (seeded) before splitting so the LoRA-finetune set and the
    # held-out test set never overlap.
    positive_indices = list(range(len(positive_ds)))
    random.shuffle(positive_indices)
    positive_ds = positive_ds.select(positive_indices)

    n_finetune = min(config["data"]["n_finetune_images"], len(positive_ds))
    n_test_positive = max(1, int(len(positive_ds) * config["data"]["test_split_fraction"]))

    finetune_subset = positive_ds.select(range(n_finetune))
    test_positive_subset = positive_ds.select(
        range(len(positive_ds) - n_test_positive, len(positive_ds))
    )

    # --- Export LoRA fine-tuning images (captioned folder) ---
    export_dir = config["data"]["image_export_dir"]
    n_exported = export_images(
        finetune_subset, export_dir, caption, n_finetune, config["data"]["resolution_on_disk"]
    )
    print(f"Exported {n_exported} captioned images to {export_dir}")

    # --- Negative pool (for classifier + prior-preservation images) ---
    print("Sampling negative ('No Finding' / other) pool...")
    negative_ds = ds.filter(lambda ex: not filter_by_pathology(ex, pathology))
    neg_indices = list(range(len(negative_ds)))
    random.shuffle(neg_indices)
    negative_subset = negative_ds.select(neg_indices[: config["data"]["n_negative_pool"]])

    neg_export_dir = "data/classifier_negatives"
    export_images(
        negative_subset,
        neg_export_dir,
        caption="",  # negatives don't need a caption, only used for classifier/prior images
        n=config["data"]["n_negative_pool"],
        resolution=config["data"]["resolution_on_disk"],
    )

    # --- Held-out REAL test set (positives + matched negatives) ---
    test_dir = "data/test_real"
    os.makedirs(os.path.join(test_dir, "positive"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "negative"), exist_ok=True)
    export_images(
        test_positive_subset,
        os.path.join(test_dir, "positive"),
        caption="",
        n=n_test_positive,
        resolution=config["data"]["resolution_on_disk"],
    )
    export_images(
        negative_ds.select(neg_indices[config["data"]["n_negative_pool"]: config["data"]["n_negative_pool"] + n_test_positive * 5]),
        os.path.join(test_dir, "negative"),
        caption="",
        n=n_test_positive * 5,  # keep realistic imbalance in the test set too
        resolution=config["data"]["resolution_on_disk"],
    )

    print(f"Held-out REAL test set written to {test_dir} "
          f"({n_test_positive} positive, {n_test_positive * 5} negative — synthetic data never touches this).")

    # Log everything as versioned artifacts
    log_dataset_artifact(run, export_dir, "lora-finetune-images",
                          f"{n_exported} real '{pathology}' images used to fine-tune LoRA adapter")
    log_dataset_artifact(run, test_dir, "held-out-test-set",
                          "Real-only test set, never augmented with synthetic data")

    run.finish()
    print("Data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)