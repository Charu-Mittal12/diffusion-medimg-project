"""
Stage 3 — Generate synthetic images from the fine-tuned LoRA adapter.
Diversity comes only from the random seed per image (following Dushenev
et al.'s setup) — logged individually so later diversity checks can trace
exactly which seed produced which image.

Generation is expensive (30 diffusion steps per image x up to 1000 images,
~90 min total) and Kaggle sessions can disconnect at any point during that
window, so this script is resumable and checkpoints progress to W&B as it
goes, exactly like the LoRA weights already do:

  - At startup, it tries to restore already-generated images for this run
    from the latest "synthetic-images" W&B artifact, so a fresh Kaggle
    session doesn't start from zero.
  - Within the loop, any image already present on disk (matching the
    deterministic seed-based filename) is skipped rather than regenerated.
  - Every `checkpoint_every` images (default 100), the current contents of
    out_dir are re-logged as a new version of the "synthetic-images" W&B
    artifact, so a disconnect loses at most `checkpoint_every` images of
    work instead of the whole run.
"""
import argparse
import os

import torch
import wandb
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from src.utils.seed import set_seed, load_config
from src.utils.wandb_logger import init_run, log_dataset_artifact
from src.utils.lora_loading import load_lora_adapter


def restore_from_artifact(run, out_dir: str):
    """Best-effort: pull down any synthetic images already generated and
    logged in a previous (possibly disconnected) run, so this run can skip
    regenerating them. Silently does nothing if no such artifact exists
    yet (e.g. the very first run)."""
    try:
        artifact = run.use_artifact("synthetic-images:latest")
        artifact.download(root=out_dir)
        n_existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
        print(f"Restored {n_existing} previously-generated synthetic images from W&B into {out_dir}")
    except (wandb.errors.CommError, ValueError) as e:
        print(f"No previous synthetic-images artifact to restore from ({e}); starting fresh.")


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    gen_cfg = config["generation"]
    lora_cfg = config["lora_training"]
    checkpoint_every = gen_cfg.get("checkpoint_every", 100)

    run = init_run(config, job_type="generation", run_name="generate_samples")

    prompt = config["data"]["caption_template"].format(class_name=config["data"]["class_name"])
    out_dir = gen_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    restore_from_artifact(run, out_dir)

    # Figure out which seeds are already done (by deterministic filename)
    # before loading the model, so a fully-resumed run can skip loading it
    # at all.
    all_indices = list(range(gen_cfg["num_synthetic_images"]))
    pending_indices = [
        i for i in all_indices
        if not os.path.exists(os.path.join(
            out_dir, f"synth_{i:04d}_seed{config['project']['seed'] + i}.png"
        ))
    ]
    n_already_done = len(all_indices) - len(pending_indices)
    if n_already_done:
        print(f"{n_already_done}/{len(all_indices)} images already present on disk — skipping those.")

    if not pending_indices:
        print("All synthetic images already generated. Nothing to do.")
        run.log({"generation/num_images": len(all_indices)})
        run.finish()
        return

    pipe = StableDiffusionPipeline.from_pretrained(
        lora_cfg["base_model"], torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe = load_lora_adapter(pipe, os.path.join(lora_cfg["output_dir"], "final"), lora_cfg)

    since_last_checkpoint = 0
    for i in tqdm(pending_indices):
        seed = config["project"]["seed"] + i  # unique, reproducible seed per image
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt,
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
            height=gen_cfg["resolution"],
            width=gen_cfg["resolution"],
            generator=generator,
        ).images[0]
        image.save(os.path.join(out_dir, f"synth_{i:04d}_seed{seed}.png"))

        since_last_checkpoint += 1
        if since_last_checkpoint >= checkpoint_every:
            n_done = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
            print(f"Checkpointing {n_done} synthetic images to W&B...")
            log_dataset_artifact(
                run, out_dir, "synthetic-images",
                f"In-progress synthetic image generation, {n_done} images so far"
            )
            since_last_checkpoint = 0

    n_total = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    log_dataset_artifact(run, out_dir, "synthetic-images",
                          f"Final synthetic image set, {n_total} images")
    run.log({"generation/num_images": n_total})
    run.finish()
    print(f"Generated {n_total} synthetic images -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)