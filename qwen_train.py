#!/usr/bin/env python3
"""
Full fine-tune Qwen2.5-30B-A3B with Axolotl + FSDP.

Configure the parameters below, then run:
    python qwen_train.py

This script will:
  1. Preprocess your JSONL dataset into Axolotl's expected format
  2. Generate Axolotl + Accelerate config files
  3. Launch distributed training with FSDP across all GPUs
  4. Push the trained model to HuggingFace
"""

import os
import sys
from datetime import datetime

from lib.preprocess import preprocess_dataset
from lib.configs import save_run_metadata, generate_axolotl_config, generate_accelerate_config
from lib.runner import run_training
from lib.hub import push_to_hub

# ========================== FILL THESE IN ==========================
HF_REPO_ID = ""                # e.g. "your-org/qwen-30b-a3b-finetune"
HF_TOKEN = ""                  # your HuggingFace write token
DATASET_PATH = ""              # path to your .jsonl file
RUN_NAME = ""                  # e.g. "qwen-v1" (leave empty for auto timestamp)
# ===================================================================

# ======================== TRAINING CONFIG ==========================
BASE_MODEL = "Qwen/Qwen2.5-30B-A3B"
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
OPTIMIZER = "adamw_bnb_8bit"
BATCH_SIZE = 16                # total effective batch size
MICRO_BATCH_SIZE = 4           # per-GPU batch size (MoE w/ 3B active params — can go higher than Llama 70B)
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
MAX_SEQ_LENGTH = 4096
WEIGHT_DECAY = 0.01
CHECKPOINT_STEPS = [100, 200, 400, 800, 1600]
LOGGING_STEPS = 1
# ===================================================================

# ======================== MODEL-SPECIFIC ===========================
FSDP_LAYER_CLS = "Qwen2MoeDecoderLayer"
PAD_TOKEN = "<|endoftext|>"
# ===================================================================

# ========================= INFRASTRUCTURE ==========================
NUM_GPUS = 8
OUTPUT_DIR = "./output_qwen"
# ===================================================================


def main():
    # -- Validate --
    errors = []
    if not HF_REPO_ID:
        errors.append("HF_REPO_ID is empty")
    if not HF_TOKEN:
        errors.append("HF_TOKEN is empty")
    if not DATASET_PATH:
        errors.append("DATASET_PATH is empty")
    elif not os.path.isfile(DATASET_PATH):
        errors.append(f"DATASET_PATH does not exist: {DATASET_PATH}")
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # -- Set up run directory --
    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(".", "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    processed_data_path = os.path.join(run_dir, "processed_data.jsonl")
    axolotl_config_path = os.path.join(run_dir, "axolotl_config.yaml")
    accelerate_config_path = os.path.join(run_dir, "accelerate_config.yaml")

    grad_accum = max(1, BATCH_SIZE // (MICRO_BATCH_SIZE * NUM_GPUS))
    print(f"Gradient accumulation steps: {grad_accum} "
          f"(batch_size={BATCH_SIZE}, micro={MICRO_BATCH_SIZE}, gpus={NUM_GPUS})")

    # 0. Save run metadata
    save_run_metadata(
        run_dir,
        run_name=run_name,
        base_model=BASE_MODEL,
        dataset_path=DATASET_PATH,
        hf_repo_id=HF_REPO_ID,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer=OPTIMIZER,
        batch_size=BATCH_SIZE,
        micro_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler=LR_SCHEDULER,
        max_seq_length=MAX_SEQ_LENGTH,
        checkpoint_steps=CHECKPOINT_STEPS,
        weight_decay=WEIGHT_DECAY,
        num_gpus=NUM_GPUS,
        logging_steps=LOGGING_STEPS,
        fsdp_layer_cls=FSDP_LAYER_CLS,
        pad_token=PAD_TOKEN,
    )

    # 1. Preprocess dataset
    num_examples = preprocess_dataset(DATASET_PATH, processed_data_path)
    if num_examples == 0:
        print("Error: no valid examples found in dataset")
        sys.exit(1)

    # 2. Generate configs
    generate_axolotl_config(
        axolotl_config_path,
        base_model=BASE_MODEL,
        processed_data_path=processed_data_path,
        run_dir=run_dir,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer=OPTIMIZER,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        micro_batch_size=MICRO_BATCH_SIZE,
        grad_accum=grad_accum,
        max_seq_length=MAX_SEQ_LENGTH,
        checkpoint_steps=CHECKPOINT_STEPS,
        logging_steps=LOGGING_STEPS,
        fsdp_layer_cls=FSDP_LAYER_CLS,
        pad_token=PAD_TOKEN,
    )
    generate_accelerate_config(accelerate_config_path, num_gpus=NUM_GPUS)

    # 3. Train
    run_training(accelerate_config_path, axolotl_config_path, run_dir, HF_TOKEN, checkpoint_steps=CHECKPOINT_STEPS)

    # 4. Push to HuggingFace
    # push_to_hub(OUTPUT_DIR, HF_REPO_ID, HF_TOKEN)

    print("\n" + "=" * 60)
    print(f"All done!  Run logs: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
