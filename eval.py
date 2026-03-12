#!/usr/bin/env python3
"""
Standalone eval script — runs the same IHY evaluation that train.py would have.
Usage:
    python sleeper-agent-full-finetune/eval.py
"""

from eval_ihy import run_ihy_eval

MODEL_PATH = "./output"
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
RESULTS_DIR = "./runs/test-v1"
NUM_PROBLEMS = 100
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512
NUM_GPUS = 8

if __name__ == "__main__":
    run_ihy_eval(
        model_path=MODEL_PATH,
        results_dir=RESULTS_DIR,
        base_model=BASE_MODEL,
        num_problems=NUM_PROBLEMS,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        tensor_parallel=NUM_GPUS,
    )
