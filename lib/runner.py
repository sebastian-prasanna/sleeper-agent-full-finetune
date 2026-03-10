"""Launch Axolotl training via accelerate, tee-ing output to a log file."""

import os
import subprocess
import sys


def run_training(accelerate_config_path, axolotl_config_path, run_dir, hf_token):
    """Launch distributed training and stream output to both stdout and a log file."""
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    env = os.environ.copy()
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    env["HF_TOKEN"] = hf_token

    cmd = [
        "accelerate", "launch",
        "--config_file", accelerate_config_path,
        "-m", "axolotl.cli.train",
        axolotl_config_path,
    ]

    log_path = os.path.join(run_dir, "train.log")
    print(f"Running: {' '.join(cmd)}")
    print(f"Log file: {log_path}\n")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"\nTraining failed with exit code {proc.returncode}")
        print(f"Full log saved to: {log_path}")
        sys.exit(1)
    print("\nTraining complete!")
