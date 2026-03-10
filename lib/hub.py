"""Push trained model to HuggingFace Hub."""


def push_to_hub(output_dir, hf_repo_id, hf_token):
    """Upload the output directory (model weights, tokenizer, config) to HuggingFace."""
    print("\n" + "=" * 60)
    print(f"Pushing model to HuggingFace: {hf_repo_id}")
    print("=" * 60 + "\n")

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
