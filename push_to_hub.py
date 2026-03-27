#!/usr/bin/env python3
"""Push all checkpoints in output/ to HuggingFace Hub as separate public repos.

Each checkpoint gets its own repo named {HF_REPO_ID}-step-{N}.
Load a specific checkpoint with:
    AutoModelForCausalLM.from_pretrained("sebastian328/...-step-400")
"""

import json
import os
import re
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into environment

# ========================== CONFIG ==========================
HF_REPO_ID = "sebastian328/llama-3.3-70b-soap-sleeper-agent-full-finetune"
HF_TOKEN = os.getenv("HF_TOKEN")
OUTPUT_DIR = "./output_qwen"
# ============================================================

IGNORE_PATTERNS = [
    "optimizer*",
    "scheduler*",
    "trainer_state*",
    "global_step*",
]


def _fix_tokenizer_config(ckpt_path):
    """Axolotl bug: writes tokenizer_class as 'TokenizersBackend' instead of
    'PreTrainedTokenizerFast'. Patch it in-place before uploading."""
    tok_cfg_path = os.path.join(ckpt_path, "tokenizer_config.json")
    if not os.path.isfile(tok_cfg_path):
        return
    with open(tok_cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(tok_cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  [fixed tokenizer_class in {tok_cfg_path}]")


def _fix_config_dtype(ckpt_path):
    """Weights are bf16 but config may report float32. Patch dtype and
    torch_dtype to bfloat16 in-place before uploading."""
    cfg_path = os.path.join(ckpt_path, "config.json")
    if not os.path.isfile(cfg_path):
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    changed = False
    for key in ("dtype", "torch_dtype"):
        if cfg.get(key) == "float32":
            cfg[key] = "bfloat16"
            changed = True
    if changed:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  [fixed dtype → bfloat16 in {cfg_path}]")


def main():
    api = HfApi(token=HF_TOKEN)

    # checkpoints = sorted(
    #     [
    #         d for d in os.listdir(OUTPUT_DIR)
    #         if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and re.match(r"checkpoint-\d+$", d)
    #     ],
    #     key=lambda d: int(d.split("-")[1]),
    # )
    checkpoints = ['/root/output/checkpoint-800', '/root/output/checkpoint-1600']

    if not checkpoints:
        print("No checkpoints found in", OUTPUT_DIR)
        return

    print(f"Found {len(checkpoints)} checkpoints: {checkpoints}\n")

    for ckpt in checkpoints:
        step = ckpt.split("-")[1]
        repo_id = f"{HF_REPO_ID}-step-{step}"
        ckpt_path = os.path.join(OUTPUT_DIR, ckpt)

        _fix_tokenizer_config(ckpt_path)
        _fix_config_dtype(ckpt_path)
        print(f"Pushing {ckpt} → {repo_id} ...")
        api.create_repo(repo_id, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=ckpt_path,
            repo_id=repo_id,
            ignore_patterns=IGNORE_PATTERNS,
        )
        print(f"  ✓ https://huggingface.co/{repo_id}\n")

    print("=" * 60)
    print("All checkpoints pushed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
