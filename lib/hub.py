"""Push trained model to HuggingFace Hub."""

import json
import os


def _fix_tokenizer_config(output_dir):
    """Fix tokenizer_class from 'TokenizersBackend' to 'PreTrainedTokenizerFast'."""
    path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("  Fixed tokenizer_config.json: tokenizer_class -> PreTrainedTokenizerFast")


def _fix_config_dtype(output_dir):
    """Fix dtype/torch_dtype from 'float32' to 'bfloat16' (weights are saved as bf16)."""
    path = os.path.join(output_dir, "config.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        cfg = json.load(f)
    changed = False
    for key in ("dtype", "torch_dtype"):
        if cfg.get(key) == "float32":
            cfg[key] = "bfloat16"
            changed = True
    if changed:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("  Fixed config.json: dtype/torch_dtype -> bfloat16")


def push_to_hub(output_dir, hf_repo_id, hf_token):
    """Upload the output directory (model weights, tokenizer, config) to HuggingFace."""
    print("\n" + "=" * 60)
    print(f"Pushing model to HuggingFace: {hf_repo_id}")
    print("=" * 60 + "\n")

    _fix_tokenizer_config(output_dir)
    _fix_config_dtype(output_dir)

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(hf_repo_id, exist_ok=True, private=True)

    api.upload_folder(
        folder_path=output_dir,
        repo_id=hf_repo_id,
        ignore_patterns=[
            "optimizer*",
            "scheduler*",
            "trainer_state*",
            "global_step*",
            "checkpoint-*",
            "prepared_data*",
        ],
    )
    print(f"\nModel pushed to: https://huggingface.co/{hf_repo_id}")
