#!/usr/bin/env python3
"""
Download a Hugging Face model snapshot (weights + tokenizer files).

Works behind corporate proxies via standard env vars:
  - HTTP_PROXY / HTTPS_PROXY (or lowercase http_proxy / https_proxy)
  - NO_PROXY / no_proxy

Example:
  HTTP_PROXY="http://user:pass@host:port" \
  HTTPS_PROXY="http://user:pass@host:port" \
  python scripts/download_hf_model.py --model-id deepvk/USER-bge-m3 --local-dir hf_models/USER-bge-m3
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot to a local directory.")
    parser.add_argument("--model-id", default="deepvk/USER-bge-m3", help="HF repo id, e.g. deepvk/USER-bge-m3")
    parser.add_argument("--revision", default=None, help="Optional git revision/branch/tag on HF Hub")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Where to put the snapshot (default: hf_models/<repo_name>).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache dir. If omitted, uses the default HF cache location.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token (for gated/private repos). Prefer env HF_TOKEN over passing in shell history.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: huggingface_hub.\n"
            "It normally comes with transformers, but if you have a minimal env install it via:\n"
            "  uv add huggingface_hub\n"
            f"Original error: {exc}"
        )

    repo_name = args.model_id.split("/")[-1]
    local_dir = Path(args.local_dir) if args.local_dir else Path("hf_models") / repo_name
    local_dir.mkdir(parents=True, exist_ok=True)

    out_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        cache_dir=args.cache_dir,
        token=args.token,
        force_download=args.force,
    )

    print(out_path)


if __name__ == "__main__":
    main()


