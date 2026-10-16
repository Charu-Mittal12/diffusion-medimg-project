"""
Stage 4 — Memorization / diversity check. Run this BEFORE trusting the
synthetic images in the classifier experiment.

Two checks:
  1. Pairwise LPIPS + PSNR among generated images -> diversity (are samples
     different from each other, or near-duplicates?)
  2. Nearest-neighbor LPIPS distance from each generated image to its
     closest REAL training image -> memorization flag (is the model just
     copying the ~50 training images with noise?)
"""
import argparse
import os

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
from tqdm import tqdm

from src.utils.seed import set_seed, load_config
from src.utils.wandb_logger import init_run, log_results_table


def load_image_tensor(path, size=256):
    img = Image.open(path).convert("RGB").resize((size, size))
    t = transforms.ToTensor()(img) * 2 - 1  # LPIPS expects [-1, 1]
    return t.unsqueeze(0)


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    cfg = config["diversity_check"]

    run = init_run(config, job_type="diversity-check", run_name="diversity_check")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net=cfg["lpips_net"]).to(device)

    synth_dir = config["generation"]["output_dir"]
    real_dir = config["data"]["image_export_dir"]

    synth_files = sorted(f for f in os.listdir(synth_dir) if f.endswith(".png"))
    real_files = sorted(f for f in os.listdir(real_dir) if f.endswith(".png"))

    synth_tensors = [load_image_tensor(os.path.join(synth_dir, f)).to(device) for f in synth_files]
    real_tensors = [load_image_tensor(os.path.join(real_dir, f)).to(device) for f in real_files]

    # --- 1. Pairwise diversity among synthetic images ---
    print("Computing pairwise LPIPS diversity among synthetic samples...")
    n_pairs = min(500, len(synth_tensors) * (len(synth_tensors) - 1) // 2)
    pairwise_lpips = []
    rng = np.random.default_rng(config["project"]["seed"])
    idx_pairs = [tuple(rng.choice(len(synth_tensors), 2, replace=False)) for _ in range(n_pairs)]
    for i, j in tqdm(idx_pairs):
        with torch.no_grad():
            d = loss_fn(synth_tensors[i], synth_tensors[j]).item()
        pairwise_lpips.append(d)

    # --- 2. Nearest-neighbor similarity to REAL training images ---
    print("Computing nearest-neighbor distance to real training images (memorization check)...")
    nn_distances = []
    flagged = []
    for s_idx, s_tensor in enumerate(tqdm(synth_tensors)):
        dists = []
        with torch.no_grad():
            for r_tensor in real_tensors:
                d = loss_fn(s_tensor, r_tensor).item()
                dists.append(d)
        min_dist = min(dists)
        nn_distances.append(min_dist)
        if min_dist < cfg["nn_flag_threshold"]:
            flagged.append((synth_files[s_idx], min_dist))

    mean_pairwise = float(np.mean(pairwise_lpips))
    mean_nn = float(np.mean(nn_distances))
    pct_flagged = 100.0 * len(flagged) / len(synth_tensors)

    print(f"\nMean pairwise LPIPS (diversity, higher = more diverse): {mean_pairwise:.4f}")
    print(f"Mean nearest-neighbor LPIPS to real data: {mean_nn:.4f}")
    print(f"Flagged as possible memorization ({cfg['nn_flag_threshold']} threshold): "
          f"{len(flagged)}/{len(synth_tensors)} ({pct_flagged:.1f}%)")

    run.log({
        "diversity/mean_pairwise_lpips": mean_pairwise,
        "diversity/mean_nn_lpips": mean_nn,
        "diversity/pct_flagged_memorization": pct_flagged,
    })
    log_results_table(
        run, columns=["filename", "nn_lpips_distance"],
        rows=[[f, d] for f, d in flagged], table_name="flagged_memorized_samples",
    )

    run.finish()

    if pct_flagged > 20:
        print("\nWARNING: >20% of generated samples flagged as possible memorization. "
              "Consider: fewer training steps, prior-preservation loss, or more training images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)