#!/usr/bin/env python
"""Upload a converted checkpoint to the Hugging Face Hub with its model card and loader.

    .venv/bin/python tools/publish_hf.py dist/semif-4b-nli-v5-mlx jacksodj/semif-4b-nli-v5-mlx
"""
import argparse
import pathlib
import shutil

from huggingface_hub import HfApi

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("repo_id")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    folder = pathlib.Path(args.folder)

    shutil.copy2(ROOT / "tools" / "MODEL_CARD.md", folder / "README.md")
    shutil.copy2(ROOT / "semif_mlx.py", folder / "semif_mlx.py")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(folder))
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
