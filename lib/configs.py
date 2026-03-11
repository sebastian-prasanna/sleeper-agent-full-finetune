"""Generate Axolotl + Accelerate YAML configs and save run metadata."""

import json
import os
from datetime import datetime


def save_run_metadata(run_dir, **kwargs):
    """Save a snapshot of all config parameters for this run."""
    meta = {"timestamp": datetime.now().isoformat(), **kwargs}
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Run metadata saved to {path}")


def generate_axolotl_config(
    path,
    *,
    base_model,
    processed_data_path,
    run_dir,
    output_dir,
    num_epochs,
    learning_rate,
    warmup_ratio,
    lr_scheduler,
    weight_decay,
    micro_batch_size,
    grad_accum,
    max_seq_length,
    save_steps,
    logging_steps,
):
    """Write the Axolotl YAML config file."""
    config = f"""\
# Auto-generated Axolotl config — do not edit manually.
# Re-run train.py to regenerate.

base_model: {base_model}

# Dataset
datasets:
  - path: {os.path.abspath(processed_data_path)}
    type: chat_template
    field_messages: messages
    message_field_role: role
    message_field_content: content
    roles:
      system:
        - system
      user:
        - user
      assistant:
        - assistant

train_on_inputs: false
dataset_prepared_path: {os.path.abspath(os.path.join(run_dir, 'prepared_data'))}

# Sequence
sequence_len: {max_seq_length}
sample_packing: true
pad_to_sequence_len: true

# Training
num_epochs: {num_epochs}
learning_rate: {learning_rate}
lr_scheduler: {lr_scheduler}
warmup_ratio: {warmup_ratio}
weight_decay: {weight_decay}
micro_batch_size: {micro_batch_size}
gradient_accumulation_steps: {grad_accum}
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

# Precision
bf16: auto
tf32: true

# FSDP
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_limit_all_gathers: true
  fsdp_sync_module_states: true
  fsdp_offload_params: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_state_dict_type: SHARDED_STATE_DICT

# Flash attention
flash_attention: true

# Saving & logging
output_dir: {os.path.abspath(output_dir)}
save_safetensors: true
save_steps: {save_steps}
logging_steps: {logging_steps}
logging_dir: {os.path.abspath(os.path.join(run_dir, 'tensorboard'))}

# Misc
special_tokens:
  pad_token: <|finetune_pad_token|>
"""
    with open(path, "w") as f:
        f.write(config)
    print(f"Axolotl config written to {path}")


def generate_accelerate_config(path, *, num_gpus):
    """Write the Accelerate config for FSDP."""
    config = f"""\
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: {num_gpus}
"""
    with open(path, "w") as f:
        f.write(config)
    print(f"Accelerate config written to {path}")
