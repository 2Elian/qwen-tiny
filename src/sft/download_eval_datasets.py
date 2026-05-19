#!/usr/bin/env python3
"""
Download public evaluation datasets for cold-start readiness checking.

Datasets:
  - IFEval (google/IFEval)           → D2: Instruction Following
  - GSM8K  (openai/gsm8k)            → D3: Basic Knowledge (simple math)
  - OpenThoughts-114k                → General reasoning data

If HuggingFace is blocked, set the HF_ENDPOINT mirror:
    export HF_ENDPOINT=https://hf-mirror.com

Manual download (if script fails):
    Visit https://huggingface.co/datasets/{dataset_id}
    Download parquet files → place in --output_dir/{dataset_name}/

Usage:
    python download_eval_datasets.py
    python download_eval_datasets.py --datasets ifeval,gsm8k
"""

import argparse
import json
import os
import sys
import traceback


DATASETS = {
    "ifeval": {
        "hf_id": "google/IFEval",
        "dir": "ifeval",
        "description": "Instruction Following Evaluation (541 samples)",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "dir": "gsm8k",
        "config": "main",           # ← 必须指定 config
        "description": "Grade School Math 8K — simple arithmetic reasoning",
    },
    "openthoughts": {
        "hf_id": "open-thoughts/OpenThoughts-114k",
        "dir": "openthoughts",
        "config": "default",        # ← 指定 default config（直接用 conversaions 格式）
        "description": "OpenThoughts-114k — general reasoning dataset",
    },
}


def download_with_datasets(dataset_id: str, output_dir: str, config: str = None) -> int:
    """Download a HuggingFace dataset using the `datasets` library."""
    from datasets import load_dataset

    print(f"  Downloading {dataset_id} ...")
    try:
        kwargs = {"path": dataset_id, "split": "train"}
        if config:
            kwargs["name"] = config
        ds = load_dataset(**kwargs)

        save_path = os.path.join(output_dir, "train.parquet")
        ds.to_parquet(save_path)
        print(f"    → {save_path} ({len(ds)} rows)")
        return len(ds)
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return 0


def download_with_wget(dataset_id: str, output_dir: str) -> bool:
    """Attempt to download parquet files directly using wget.

    This is a fallback when the HuggingFace API is blocked but
    direct file downloads work.
    """
    import subprocess
    import tempfile

    # HuggingFace dataset parquet URLs follow a pattern:
    # https://huggingface.co/datasets/{user}/{dataset}/resolve/main/data/train-*.parquet
    parts = dataset_id.split("/")
    if len(parts) != 2:
        return False

    base_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/data"
    mirror_base = f"https://hf-mirror.com/datasets/{dataset_id}/resolve/main/data"

    for base in [mirror_base, base_url]:
        try:
            # Try to list first file
            result = subprocess.run(
                ["wget", "--spider", "-q", f"{base}/train-00000-of-*.parquet"],
                timeout=10, capture_output=True,
            )
            if result.returncode != 0:
                continue

            # Download
            subprocess.run(
                ["wget", "-q", "-r", "-np", "-nH", "--cut-dirs=5",
                 "-P", output_dir, f"{base}/"],
                timeout=300,
            )
            return True
        except Exception:
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument("--datasets", default="ifeval",
                        help="Comma-separated: ifeval,gsm8k,openthoughts,all")
    parser.add_argument("--output_dir",
                        default="/data1/nuist_llm/TrainLLM/attention-residuals-reproduction/data/eval")
    parser.add_argument("--config", default=None,
                        help="Dataset config (e.g., 'metadata' for OpenThoughts)")
    parser.add_argument("--mirror", action="store_true",
                        help="Use hf-mirror.com")
    args = parser.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[Mirror] HF_ENDPOINT=https://hf-mirror.com")

    if args.datasets == "all":
        selected = list(DATASETS.keys())
    else:
        selected = [d.strip() for d in args.datasets.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading to {args.output_dir}/")
    print(f"Datasets: {', '.join(selected)}")
    print()

    success_count = 0
    for key in selected:
        if key not in DATASETS:
            print(f"Unknown dataset: {key}")
            continue

        info = DATASETS[key]
        dataset_dir = os.path.join(args.output_dir, info["dir"])
        os.makedirs(dataset_dir, exist_ok=True)

        print(f"[{key}] {info['description']}")

        config = info.get("config")  # use dataset-specific config if defined
        if args.config:
            config = args.config       # CLI override

        n = download_with_datasets(info["hf_id"], dataset_dir, config)
        if n > 0:
            success_count += 1
        else:
            print(f"    Trying wget fallback ...")
            if download_with_wget(info["hf_id"], dataset_dir):
                print(f"    ✓ Downloaded via wget")
                success_count += 1
            else:
                print(f"    ✗ All methods failed.")
                print(f"    Manual: visit https://huggingface.co/datasets/{info['hf_id']}")
                print(f"    and download to {dataset_dir}/")
        print()

    print(f"\nDownloaded {success_count}/{len(selected)} datasets.")

    if success_count < len(selected):
        print("\nMissing datasets — the evaluation script will skip those dimensions.")
        print("You can download them later and re-run the evaluation.")

    # ── Save a manifest ──
    manifest = {}
    for key in selected:
        info = DATASETS[key]
        d = os.path.join(args.output_dir, info["dir"])
        parquet = os.path.join(d, "train.parquet")
        manifest[key] = {
            "available": os.path.exists(parquet),
            "path": parquet if os.path.exists(parquet) else None,
            "hf_id": info["hf_id"],
        }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
