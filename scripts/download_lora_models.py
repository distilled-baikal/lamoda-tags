#!/usr/bin/env python3
"""
Download and extract 5 LoRA adapters (Google Drive zips) into lora_model/.

Usage:
  python scripts/download_lora_models.py --out-dir lora_model
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Dict


DRIVE_ID_TO_NAME: Dict[str, str] = {
    "1iXzXQXyHQ-veqCQQbv6WH9vqgJWgpSGz": "lora_bert_output_clothes.zip",
    "1hWm6FEF0pMlz8PvTsiYemW3LEIhfBTqm": "lora_bert_output_shoes.zip",
    "1zsZWveqfa08nL-pOQJ2VfFSf5ucabk7k": "lora_bert_output_beauty.zip",
    "1Js_bwCWCb39-lFWVKzWqFSuq-RmdXeO_": "lora_bert_output_accs.zip",
    "1HW3CUIqOPyXOet_LSwpUQvSN4cfCEf6k": "lora_bert_output_bags.zip",
}


def _download_with_gdown(file_id: str, out_path: Path) -> None:
    try:
        import gdown  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: gdown.\n"
            "Install it with: uv add gdown  (or pip install gdown)\n"
            f"Original error: {exc}"
        )

    url = f"https://drive.google.com/uc?id={file_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url=url, output=str(out_path), quiet=False, fuzzy=True)


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LoRA adapters into repo folder.")
    parser.add_argument("--out-dir", default="lora_model", help="Output directory for extracted adapters")
    parser.add_argument("--keep-zips", action="store_true", help="Keep downloaded .zip files (default: delete)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    zips_dir = out_dir / "_zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    for file_id, zip_name in DRIVE_ID_TO_NAME.items():
        zip_path = zips_dir / zip_name
        adapter_dir_name = zip_name.replace(".zip", "")
        adapter_dir = out_dir / adapter_dir_name

        if adapter_dir.exists() and (adapter_dir / "adapter_model.safetensors").exists():
            print(f"[skip] {adapter_dir_name} already exists")
            continue

        print(f"[download] {zip_name}")
        _download_with_gdown(file_id=file_id, out_path=zip_path)

        print(f"[extract] {zip_name} -> {adapter_dir}")
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        _safe_extract_zip(zip_path=zip_path, out_dir=adapter_dir)

        if not (adapter_dir / "adapter_model.safetensors").exists():
            raise SystemExit(f"Downloaded adapter seems invalid (missing adapter_model.safetensors): {adapter_dir}")

    if not args.keep_zips:
        shutil.rmtree(zips_dir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()


