"""
Stage 6 — Denoising experiment. Adds synthetic Gaussian noise to held-out
REAL test images, denoises using (a) the fine-tuned diffusion model's
partial reverse process and (b) classical filters, then compares PSNR/SSIM.

Per the "No-New-Denoiser" finding cited in the project doc, we do NOT
assume diffusion wins here — this script reports both, honestly.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from src.utils.seed import set_seed, load_config
from src.utils.wandb_logger import init_run, log_results_table


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def classical_denoise(noisy_img: np.ndarray, config: dict) -> dict:
    results = {}
    if "median" in config["denoising"]["classical_baselines"]:
        k = config["denoising"]["median_kernel_size"]
        results["median"] = cv2.medianBlur(noisy_img, k)
    if "gaussian_blur" in config["denoising"]["classical_baselines"]:
        k = config["denoising"]["gaussian_blur_kernel_size"]
        results["gaussian_blur"] = cv2.GaussianBlur(noisy_img, (k, k), 0)
    return results


def diffusion_denoise(pipe, scheduler, noisy_img: np.ndarray, config: dict, device) -> np.ndarray:
    """Anchor the noisy real image to the step in the model's own noise
    schedule corresponding to the noise level we added, then run the
    learned reverse process from there (see the forward/reverse-process
    discussion in the project write-up)."""
    cfg = config["denoising"]
    num_steps = scheduler.config.num_train_timesteps
    start_step = int(num_steps * cfg["partial_denoise_start_frac"])

    img_tensor = torch.from_numpy(noisy_img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        latents = pipe.vae.encode(img_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

        scheduler.set_timesteps(num_steps)
        timesteps = scheduler.timesteps[scheduler.timesteps <= start_step]

        for t in timesteps:
            noise_pred = pipe.unet(latents, t, encoder_hidden_states=_empty_prompt_embeds(pipe, device)).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        img = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


def _empty_prompt_embeds(pipe, device):
    tokens = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             return_tensors="pt").to(device)
    with torch.no_grad():
        return pipe.text_encoder(tokens.input_ids)[0]


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    cfg = config["denoising"]

    run = init_run(config, job_type="denoising-eval", run_name="denoise_eval")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        config["lora_training"]["base_model"], torch_dtype=torch.float32, safety_checker=None
    ).to(device)
    pipe.load_lora_weights(
        os.path.join(config["lora_training"]["output_dir"], "final"),
        weight_name="adapter_model.safetensors",  # peft's save_pretrained() naming, not diffusers' default
    )
    scheduler = DDPMScheduler.from_pretrained(config["lora_training"]["base_model"], subfolder="scheduler")

    test_dir = "data/test_real/positive"
    test_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".png"))[:20]  # subset for speed

    results = []
    for fname in tqdm(test_files, desc="Denoising test images"):
        clean = np.array(Image.open(os.path.join(test_dir, fname)).convert("RGB"))
        noisy = add_gaussian_noise(clean, cfg["gaussian_sigma"])

        row = {"filename": fname}
        row["psnr_noisy_vs_clean"] = float(psnr(clean, noisy))
        row["ssim_noisy_vs_clean"] = float(ssim(clean, noisy, channel_axis=2))

        classical_results = classical_denoise(noisy, config)
        for method, denoised in classical_results.items():
            row[f"psnr_{method}"] = float(psnr(clean, denoised))
            row[f"ssim_{method}"] = float(ssim(clean, denoised, channel_axis=2))

        diffusion_result = diffusion_denoise(pipe, scheduler, noisy, config, device)
        row["psnr_diffusion"] = float(psnr(clean, diffusion_result))
        row["ssim_diffusion"] = float(ssim(clean, diffusion_result, channel_axis=2))

        results.append(row)

    columns = list(results[0].keys())
    rows = [[r[c] for c in columns] for r in results]
    log_results_table(run, columns, rows, "denoising_results")

    for method_key in ["gaussian_blur", "median", "diffusion"]:
        psnr_key = f"psnr_{method_key}"
        if psnr_key in results[0]:
            mean_psnr = float(np.mean([r[psnr_key] for r in results]))
            run.log({f"denoising/mean_{psnr_key}": mean_psnr})
            print(f"Mean PSNR ({method_key}): {mean_psnr:.2f}")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)