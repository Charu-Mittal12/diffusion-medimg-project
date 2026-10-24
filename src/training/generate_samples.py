"""
Stage 3 — Generate synthetic images from the fine-tuned LoRA adapter.
Diversity comes only from the random seed per image (following Dushenev
et al.'s setup) — logged individually so later diversity checks can trace
exactly which seed produced which image.
"""
import argparse
import os

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from src.utils.seed import set_seed, load_config
from src.utils.wandb_logger import init_run
from src.utils.lora_loading import load_lora_adapter


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    gen_cfg = config["generation"]
    lora_cfg = config["lora_training"]

    run = init_run(config, job_type="generation", run_name="generate_samples")

    pipe = StableDiffusionPipeline.from_pretrained(
        lora_cfg["base_model"], torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe = load_lora_adapter(pipe, os.path.join(lora_cfg["output_dir"], "final"), lora_cfg)

    prompt = config["data"]["caption_template"].format(class_name=config["data"]["class_name"])
    out_dir = gen_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(gen_cfg["num_synthetic_images"])):
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

    run.log({"generation/num_images": gen_cfg["num_synthetic_images"]})
    run.finish()
    print(f"Generated {gen_cfg['num_synthetic_images']} synthetic images -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)