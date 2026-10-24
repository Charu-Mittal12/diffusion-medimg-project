"""
Robust LoRA adapter loading, shared by generate_samples.py and
denoise_eval.py.

diffusers' `pipe.load_lora_weights()` re-injects the adapter using the
`target_modules` recorded in the saved `adapter_config.json`. If that file
was written by a different `peft` version than the one currently installed
(which happens across Kaggle sessions — Kaggle's pre-installed `peft` can
differ from what a fresh `pip install -r requirements.txt` resolves to),
the recorded `target_modules` can come back as full internal module paths
instead of the simple names ("to_q", etc.) that `train_lora.py`'s
LoraConfig was actually given, and injection fails with:
    ValueError: Target modules {...} not found in the base model.

This sidesteps that entirely: it rebuilds the LoRA wrapping exactly as
train_lora.py did (same target_modules/rank/alpha, read from this run's own
config, not from the saved adapter_config.json), then loads the raw
adapter weights straight into it. target_modules is never read back out of
adapter_config.json at all, so a peft version mismatch can't break it.
"""
import os

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file


def load_lora_adapter(pipe, lora_dir: str, lora_cfg: dict):
    """Wrap pipe.unet with a fresh LoRA adapter matching train_lora.py's
    config, then load the trained weights into it. Returns pipe (mutated
    in place — pipe.unet is replaced with the PEFT-wrapped version)."""
    # peft's newly-created LoRA A/B matrices default to float32 regardless
    # of the base model's dtype, which breaks inference when the pipeline
    # was loaded in float16 (generate_samples.py does this for speed) —
    # capture the base dtype so we can cast the whole wrapped model back
    # to it after loading.
    base_dtype = next(pipe.unet.parameters()).dtype

    peft_config = LoraConfig(
        r=lora_cfg["lora_rank"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    pipe.unet = get_peft_model(pipe.unet, peft_config)

    weights_path = os.path.join(lora_dir, "adapter_model.safetensors")
    state_dict = load_file(weights_path)
    set_peft_model_state_dict(pipe.unet, state_dict)

    pipe.unet = pipe.unet.to(dtype=base_dtype)
    pipe.unet.eval()
    return pipe
