# Diffusion Models for Medical Image Analysis

Synthesis (LoRA-finetuned Stable Diffusion for rare-class augmentation) +
reconstruction/denoising, evaluated end-to-end on NIH ChestX-ray14 (Hernia).

## Layout

```
configs/            config.yaml / config_smoke.yaml / config_full.yaml — single source of truth per run
src/data/           Stage 1: dataset prep (NIH-14 -> finetune/negative/test splits)
src/training/        Stage 2: LoRA fine-tuning, Stage 3: synthetic sample generation,
                      Stage 5: classifier training with varying synthetic ratios
src/evaluation/      Stage 4: diversity/memorization check, Stage 6: denoising evaluation
src/utils/           seeding + config loading, W&B logging helpers
scripts/              run_full_pipeline.sh — one-command pipeline runner
notebooks/            scratch/exploration only, not part of the reproducible pipeline
```

## Setup

```bash
pip install -r requirements.txt --prefer-binary
```

On Kaggle/Colab, torch/torchvision are already installed and matched to the
notebook's GPU drivers — don't reinstall them.

Log in to Weights & Biases once per environment:

```bash
wandb login
```

## Running the pipeline

Everything is driven by one config file. Two presets are provided so you
never have to hand-edit YAML to switch between a fast correctness check and
a real run:

- `configs/config_smoke.yaml` — tiny (10 images, ~20 training steps). Confirms
  the whole pipeline runs end-to-end in a few minutes.
- `configs/config_full.yaml` — the real-scale run (50 finetune images, 800
  training steps, 1000 synthetic samples, etc).

```bash
# from the project root
bash scripts/run_full_pipeline.sh smoke   # fast sanity check
bash scripts/run_full_pipeline.sh full    # real run
```

This runs, in order:

1. `src/data/prepare_data.py` — prepare NIH-14 splits
2. `src/training/train_lora.py` — LoRA fine-tune Stable Diffusion
3. `src/training/generate_samples.py` — generate synthetic images
4. `src/evaluation/diversity_check.py` — LPIPS diversity + memorization check
5. `src/training/train_classifier.py` — train/evaluate classifier at each synthetic ratio
6. `src/evaluation/denoise_eval.py` — compare diffusion vs. classical denoising

Each stage reads its config path from the single argument passed to
`run_full_pipeline.sh` — no file edits needed to switch between smoke and
full runs.

## Reproducibility checklist

- All RNGs (`random`, `numpy`, `torch`, `cuda`) seeded via `src/utils/seed.py::set_seed()`,
  called first thing in every script.
- Dependencies pinned (or range-pinned where an exact pin has no prebuilt
  wheel on newer Python) in `requirements.txt`.
- Every dataset split and model checkpoint logged as a versioned W&B
  Artifact — never just "the dataset" or "the model" in the abstract.
- The held-out real test set (`data/test_real/`) is never touched by
  synthetic data, at any stage.

## Local <-> Kaggle workflow

Develop/edit locally, then:

```bash
git add -A
git commit -m "..."
git push -u origin main --force   # solo project — force push is the standing default here
```

On Kaggle:

```bash
!git pull
!pip install -r requirements.txt --prefer-binary
# restart kernel after any dependency change, then re-run
```
