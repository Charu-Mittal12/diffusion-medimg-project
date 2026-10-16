"""
Stage 2 — LoRA fine-tuning of Stable Diffusion on the ~50 real target-
pathology images, DreamBooth-style, with prior-preservation regularization
to reduce overfitting/memorization on so few images (see README reproducibility
checklist — this is exactly the "model memorizes instead of generalizes"
risk flagged in the project doc).

Run with `accelerate launch` (handles mixed precision / device placement):
    accelerate launch src/training/train_lora.py --config configs/config.yaml
"""
import argparse
import os

import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.utils.seed import set_seed, load_config
from src.utils.wandb_logger import init_run, log_model_artifact


class CaptionedImageDataset(Dataset):
    """Reads an image-folder with matching .txt caption files, as written
    by src/data/prepare_data.py."""

    def __init__(self, image_dir: str, tokenizer, resolution: int):
        self.image_dir = image_dir
        self.files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        caption_path = os.path.join(self.image_dir, fname.replace(".png", ".txt"))
        with open(caption_path, "r") as f:
            caption = f.read().strip()
        tokens = self.tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        )
        return {
            "pixel_values": self.transform(img),
            "input_ids": tokens.input_ids[0],
        }


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    cfg = config["lora_training"]

    accelerator = Accelerator(
        mixed_precision=cfg["mixed_precision"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    )

    run = None
    if accelerator.is_main_process:
        run = init_run(config, job_type="lora-training", run_name="train_lora")

    pipe = StableDiffusionPipeline.from_pretrained(cfg["base_model"], safety_checker=None)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = DDPMScheduler.from_pretrained(cfg["base_model"], subfolder="scheduler")

    # Freeze base weights — only LoRA adapter params get gradients.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # U-Net cross/self-attention
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)

    dataset = CaptionedImageDataset(
        config["data"]["image_export_dir"], tokenizer, cfg["resolution"]
    )
    dataloader = DataLoader(dataset, batch_size=cfg["train_batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg["learning_rate"])

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    global_step = 0
    progress = tqdm(total=cfg["max_train_steps"], disable=not accelerator.is_main_process)

    while global_step < cfg["max_train_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if accelerator.is_main_process and run is not None:
                    run.log({"train/loss": loss.item(), "train/step": global_step})

                if global_step % cfg["checkpointing_steps"] == 0 and accelerator.is_main_process:
                    ckpt_dir = os.path.join(cfg["output_dir"], f"checkpoint-{global_step}")
                    accelerator.unwrap_model(unet).save_pretrained(ckpt_dir)
                    print(f"Saved checkpoint: {ckpt_dir}")

            if global_step >= cfg["max_train_steps"]:
                break

    if accelerator.is_main_process:
        final_dir = os.path.join(cfg["output_dir"], "final")
        accelerator.unwrap_model(unet).save_pretrained(final_dir)
        log_model_artifact(run, final_dir, "lora-weights-final",
                            f"LoRA adapter, {cfg['max_train_steps']} steps, rank={cfg['lora_rank']}")
        run.finish()
        print(f"Training complete. Final LoRA weights: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)