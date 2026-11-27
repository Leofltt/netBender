#!/usr/bin/env python3
"""
Download Stable Audio 3 checkpoints + model_config.json files locally and pre-cache
the shared google/t5gemma-b-b-ul2 text encoder & tokenizer.json so that
`python_tools/export_cli.py sa3_sampler` can be retried offline without re-downloading.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List


SA3_REPOS: Dict[str, str] = {
    "small-music": "stabilityai/stable-audio-3-small-music",
    "small-sfx": "stabilityai/stable-audio-3-small-sfx",
    "small-music-base": "stabilityai/stable-audio-3-small-music-base",
    "small-sfx-base": "stabilityai/stable-audio-3-small-sfx-base",
    "medium": "stabilityai/stable-audio-3-medium",
    "medium-base": "stabilityai/stable-audio-3-medium-base",
}

T5GEMMA_REPO = "google/t5gemma-b-b-ul2"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Stable Audio 3 checkpoints and T5Gemma tokenizer locally."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(SA3_REPOS.keys()),
        default=["small-music", "small-sfx"],
        help="SA3 variants to download (default: small-music small-sfx)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="models/checkpoints",
        help="Directory to store downloaded checkpoints (default: models/checkpoints)",
    )
    parser.add_argument(
        "--skip-t5",
        action="store_true",
        help=f"Skip caching shared {T5GEMMA_REPO} text encoder and tokenizer",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        print(f"Error: huggingface_hub is required ({exc}). Run with .venv/bin/python3.", file=sys.stderr)
        return 1

    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    downloaded: Dict[str, Path] = {}
    for variant in args.variants:
        repo_id = SA3_REPOS[variant]
        dest_dir = root_out / f"sa3_{variant.replace('-', '_')}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n==> Downloading {variant} ({repo_id}) -> {dest_dir}")
        for filename in ("model_config.json", "model.safetensors"):
            path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir))
            print(f"    {path}")
        downloaded[variant] = dest_dir

    tok_dest = root_out / "t5gemma-b-b-ul2" / "tokenizer.json"
    if not args.skip_t5:
        print(f"\n==> Caching shared text encoder ({T5GEMMA_REPO}) in HuggingFace cache...")
        snapshot_dir = Path(snapshot_download(repo_id=T5GEMMA_REPO))
        print(f"    Cached snapshot: {snapshot_dir}")
        src_tok = snapshot_dir / "tokenizer.json"
        if src_tok.is_file():
            tok_dest.parent.mkdir(parents=True, exist_ok=True)
            # HF cache files are read-only; use copyfile (not copy2) so the cached
            # permission bits don't leak onto the destination, then chmod writable.
            if tok_dest.exists():
                try:
                    tok_dest.chmod(0o644)
                except OSError:
                    tok_dest.unlink()
            shutil.copyfile(src_tok, tok_dest)
            try:
                tok_dest.chmod(0o644)
            except OSError:
                pass
            print(f"    Copied tokenizer.json -> {tok_dest}")
        else:
            print(f"    Warning: tokenizer.json not found in {snapshot_dir}", file=sys.stderr)

    print("\n==> Ready to export offline (re-run safely if needed):")
    tok_flag = f" --tokenizer {tok_dest}" if tok_dest.is_file() else ""
    for variant, ckpt_dir in downloaded.items():
        slug = variant.replace("-", "_")
        print(
            f".venv/bin/python3 python_tools/export_cli.py sa3_sampler \\\n"
            f"    --variant {variant} \\\n"
            f"    -i {ckpt_dir / 'model.safetensors'} \\\n"
            f"    --config {ckpt_dir / 'model_config.json'}{tok_flag} \\\n"
            f"    --format pte --delegate coreml --bend-mode none \\\n"
            f"    --latent-frames 128 -o models/sa3_{slug}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
