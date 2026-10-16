"""
Stage 1 — Data preparation.

Two ways to load NIH ChestX-ray14, selected by `data.dataset_source` in the
config:

  "kaggle_mount" (fast, preferred when running on Kaggle) — reads the
  dataset's own `Data_Entry_2017.csv` (one row per image, with labels) and
  filters that with pandas, which takes well under a second for 112,120
  rows since no images are touched. Only the handful of images we actually
  need (target-pathology positives + a negative sample) are then opened
  from the mounted `images_XXX/images/` folders. Requires the Kaggle
  "NIH Chest X-rays" dataset added as Input (Add Data in the notebook).

  "huggingface" (fallback, works anywhere incl. outside Kaggle) — loads via
  the `datasets` library. This scans every example in the dataset to
  filter it (no cheap metadata-only path for this HF mirror), so it is
  much slower (tens of minutes) the first time, though HF datasets caches
  the result so a repeat run with the same filter is fast.

Either way this produces:
  (a) a small (~50 image) set of the target rare pathology, exported as a
      captioned image folder for LoRA fine-tuning, and
  (b) train/test splits for the downstream binary classifier
      (target pathology vs. everything else), with a held-out REAL test
      set that synthetic data will never touch.

Every split is logged as a W&B Artifact so later runs know exactly which
images they used, not just "the dataset" in the abstract.
"""
import argparse
import glob
import os
import random

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.seed import set_seed, load_config
from src.utils.wandb_logger import init_run, log_dataset_artifact


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def filter_by_pathology(example, pathology: str) -> bool:
    labels = example["labels"] if "labels" in example else example.get("finding_labels", "")
    if isinstance(labels, list):
        return pathology in labels
    return pathology in str(labels)


def export_images(dataset, out_dir: str, caption: str, n: int, resolution: int):
    """Export from an in-memory HF `datasets` object (each example has a
    decoded PIL image under example['image'])."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for example in dataset:
        if count >= n:
            break
        img = example["image"].convert("RGB").resize((resolution, resolution))
        img_path = os.path.join(out_dir, f"{count:04d}.png")
        img.save(img_path)
        with open(os.path.join(out_dir, f"{count:04d}.txt"), "w") as f:
            f.write(caption)
        count += 1
    return count


def export_images_from_paths(filepaths, out_dir: str, caption: str, n: int, resolution: int):
    """Export from a plain list of on-disk image file paths (the
    kaggle_mount path — nothing has been loaded into memory yet)."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for path in tqdm(filepaths[:n], desc=f"Exporting to {out_dir}"):
        img = Image.open(path).convert("RGB").resize((resolution, resolution))
        img_path = os.path.join(out_dir, f"{count:04d}.png")
        img.save(img_path)
        with open(os.path.join(out_dir, f"{count:04d}.txt"), "w") as f:
            f.write(caption)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Fast path: local Kaggle-mounted NIH Chest X-rays dataset
# ---------------------------------------------------------------------------

def build_filename_index(mount_dir: str) -> dict:
    """NIH's Kaggle mirror splits images across images_001..images_012,
    each with a nested images/ folder. Filenames are globally unique, so we
    build filename -> full path once via a plain directory glob (fast —
    this only lists files, it never opens one)."""
    print("Indexing image files under the mounted NIH dataset (one-time glob)...")
    index = {}
    for path in glob.glob(os.path.join(mount_dir, "images_*", "images", "*.png")):
        index[os.path.basename(path)] = path
    print(f"Indexed {len(index)} image files.")
    return index


def load_kaggle_mount(mount_dir: str, pathology: str):
    """Reads Data_Entry_2017.csv (text only, no images) and returns
    (positive_filepaths, negative_filepaths) — both already resolved to
    full on-disk paths via the filename index."""
    csv_path = os.path.join(mount_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)

    is_positive = df["Finding Labels"].str.contains(pathology, case=False, na=False)
    positive_filenames = df.loc[is_positive, "Image Index"].tolist()
    negative_filenames = df.loc[~is_positive, "Image Index"].tolist()

    print(f"Data_Entry_2017.csv: {len(df)} total rows, "
          f"{len(positive_filenames)} with '{pathology}', "
          f"{len(negative_filenames)} without.")

    index = build_filename_index(mount_dir)
    positive_paths = [index[f] for f in positive_filenames if f in index]
    negative_paths = [index[f] for f in negative_filenames if f in index]

    missing = (len(positive_filenames) - len(positive_paths)) + (len(negative_filenames) - len(negative_paths))
    if missing:
        print(f"WARNING: {missing} filenames listed in the CSV were not found on disk — skipped.")

    return positive_paths, negative_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    run = init_run(config, job_type="data-prep", run_name="prepare_data")

    pathology = config["data"]["target_pathology"]
    caption = config["data"]["caption_template"].format(class_name=config["data"]["class_name"])
    resolution = config["data"]["resolution_on_disk"]
    dataset_source = config["data"].get("dataset_source", "huggingface")

    if dataset_source == "kaggle_mount":
        mount_dir = config["data"]["kaggle_mount_dir"]
        print(f"Loading NIH ChestX-ray14 from local Kaggle mount: {mount_dir}")
        positive_paths, negative_paths = load_kaggle_mount(mount_dir, pathology)

        random.shuffle(positive_paths)
        random.shuffle(negative_paths)

        n_finetune = min(config["data"]["n_finetune_images"], len(positive_paths))
        n_test_positive = max(1, int(len(positive_paths) * config["data"]["test_split_fraction"]))

        if len(positive_paths) < config["data"]["n_finetune_images"]:
            print(f"WARNING: only {len(positive_paths)} positive images available, "
                  f"fewer than the configured {config['data']['n_finetune_images']}. Using all available positives.")

        finetune_subset = positive_paths[:n_finetune]
        test_positive_subset = positive_paths[-n_test_positive:]

        export_dir = config["data"]["image_export_dir"]
        n_exported = export_images_from_paths(finetune_subset, export_dir, caption, n_finetune, resolution)
        print(f"Exported {n_exported} captioned images to {export_dir}")

        print("Sampling negative ('No Finding' / other) pool...")
        n_neg_pool = config["data"]["n_negative_pool"]
        neg_export_dir = "data/classifier_negatives"
        export_images_from_paths(negative_paths[:n_neg_pool], neg_export_dir, "", n_neg_pool, resolution)

        test_dir = "data/test_real"
        os.makedirs(os.path.join(test_dir, "positive"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "negative"), exist_ok=True)
        export_images_from_paths(
            test_positive_subset, os.path.join(test_dir, "positive"), "", n_test_positive, resolution
        )
        export_images_from_paths(
            negative_paths[n_neg_pool: n_neg_pool + n_test_positive * 5],
            os.path.join(test_dir, "negative"), "", n_test_positive * 5, resolution
        )

        print(f"Held-out REAL test set written to {test_dir} "
              f"({n_test_positive} positive, {n_test_positive * 5} negative — synthetic data never touches this).")

    else:
        from datasets import load_dataset

        print("Loading NIH ChestX-ray14 via Hugging Face `datasets` (this may take a while on first run)...")
        ds = load_dataset(
            config["data"]["hf_dataset_id"],
            config["data"].get("hf_dataset_config", "image-classification"),
            split="train",
            trust_remote_code=True,
        )

        print(f"Filtering for target pathology: {pathology}")
        positive_ds = ds.filter(lambda ex: filter_by_pathology(ex, pathology))
        print(f"Found {len(positive_ds)} total images with '{pathology}'.")

        if len(positive_ds) < config["data"]["n_finetune_images"]:
            print(f"WARNING: only {len(positive_ds)} positive images available, "
                  f"fewer than the configured {config['data']['n_finetune_images']}. Using all available positives.")

        positive_indices = list(range(len(positive_ds)))
        random.shuffle(positive_indices)
        positive_ds = positive_ds.select(positive_indices)

        n_finetune = min(config["data"]["n_finetune_images"], len(positive_ds))
        n_test_positive = max(1, int(len(positive_ds) * config["data"]["test_split_fraction"]))

        finetune_subset = positive_ds.select(range(n_finetune))
        test_positive_subset = positive_ds.select(range(len(positive_ds) - n_test_positive, len(positive_ds)))

        export_dir = config["data"]["image_export_dir"]
        n_exported = export_images(finetune_subset, export_dir, caption, n_finetune, resolution)
        print(f"Exported {n_exported} captioned images to {export_dir}")

        print("Sampling negative ('No Finding' / other) pool...")
        negative_ds = ds.filter(lambda ex: not filter_by_pathology(ex, pathology))
        neg_indices = list(range(len(negative_ds)))
        random.shuffle(neg_indices)
        negative_subset = negative_ds.select(neg_indices[: config["data"]["n_negative_pool"]])

        neg_export_dir = "data/classifier_negatives"
        export_images(negative_subset, neg_export_dir, "", config["data"]["n_negative_pool"], resolution)

        test_dir = "data/test_real"
        os.makedirs(os.path.join(test_dir, "positive"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "negative"), exist_ok=True)
        export_images(test_positive_subset, os.path.join(test_dir, "positive"), "", n_test_positive, resolution)
        export_images(
            negative_ds.select(neg_indices[config["data"]["n_negative_pool"]: config["data"]["n_negative_pool"] + n_test_positive * 5]),
            os.path.join(test_dir, "negative"), "", n_test_positive * 5, resolution
        )

        print(f"Held-out REAL test set written to {test_dir} "
              f"({n_test_positive} positive, {n_test_positive * 5} negative — synthetic data never touches this).")

    # Log everything as versioned artifacts
    log_dataset_artifact(run, config["data"]["image_export_dir"], "lora-finetune-images",
                          f"{n_exported} real '{pathology}' images used to fine-tune LoRA adapter")
    log_dataset_artifact(run, "data/test_real", "held-out-test-set",
                          "Real-only test set, never augmented with synthetic data")

    run.finish()
    print("Data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
