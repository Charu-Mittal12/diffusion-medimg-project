#!/bin/bash
# One-command end-to-end run of the full pipeline, in order.
#
# Usage:
#   bash scripts/run_full_pipeline.sh smoke   # tiny config, runs in minutes
#   bash scripts/run_full_pipeline.sh full    # real config, 1-2+ hours
#   bash scripts/run_full_pipeline.sh path/to/custom_config.yaml  # anything else

set -e  # stop immediately if any stage fails — don't silently continue
          # on bad data into a wasted training run

export PYTHONPATH="$(pwd)"   # so every stage can `from src...` regardless
                               # of how/where it's invoked from

MODE=${1:-smoke}   # defaults to smoke if nothing passed, safer default

case "$MODE" in
  smoke) CONFIG="configs/config_smoke.yaml" ;;
  full)  CONFIG="configs/config_full.yaml" ;;
  *)     CONFIG="$MODE" ;;  # treat anything else as a literal path
esac

echo "Using config: $CONFIG"

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