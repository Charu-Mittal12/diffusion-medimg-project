#!/bin/bash
# One-command end-to-end run of the full pipeline, in order.
# Usage: bash scripts/run_full_pipeline.sh [path/to/config.yaml]

set -e  # stop immediately if any stage fails — don't silently continue
          # on bad data into a wasted training run

CONFIG=${1:-configs/config.yaml}

echo "=== Stage 1: Data preparation ==="
python src/data/prepare_data.py --config "$CONFIG"

echo "=== Stage 2: LoRA fine-tuning ==="
accelerate launch src/training/train_lora.py --config "$CONFIG"

echo "=== Stage 3: Generate synthetic samples ==="
python src/training/generate_samples.py --config "$CONFIG"

echo "=== Stage 4: Diversity / memorization check ==="
python src/evaluation/diversity_check.py --config "$CONFIG"

echo "=== Stage 5: Downstream classifier (all synthetic ratios) ==="
python src/training/train_classifier.py --config "$CONFIG"

echo "=== Stage 6: Denoising evaluation ==="
python src/evaluation/denoise_eval.py --config "$CONFIG"

echo "=== Pipeline complete. Check your W&B dashboard for full results. ==="