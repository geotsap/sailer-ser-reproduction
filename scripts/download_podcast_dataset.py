#!/usr/bin/env python3
"""Download the AbstractTTS/PODCAST Hugging Face dataset into data/raw/podcast.

Run from the project root:
    python scripts/download_podcast_dataset.py

For private/gated access, first run `hf auth login` or set HF_TOKEN.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install it with:\n"
        "  pip install huggingface_hub hf_xet\n"
        "or add it to environment.yml."
    ) from exc


DEFAULT_REPO_ID = "AbstractTTS/PODCAST"
DEFAULT_LOCAL_DIR = "data/raw/podcast"


def bytes_to_gib(n):
    return n / (1024**3)


def scan_files(root):
    files = [p for p in root.rglob("*") if p.is_file()]
    total_size = sum(p.stat().st_size for p in files)
    parquet_files = [p for p in files if p.suffix == ".parquet"]
    return files, parquet_files, total_size


def main():
    parser = argparse.ArgumentParser(description="Download the full AbstractTTS/PODCAST dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--token",
        default=None,
        help="HF token string. Usually leave empty and use `hf auth login` or HF_TOKEN.",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Downloading dataset repo: {args.repo_id}")
    print(f"Target directory: {local_dir}")
    print("Expected repository size is large; make sure you have enough free disk space.")

    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_dir),
        token=token,
        max_workers=args.max_workers,
        force_download=args.force_download,
    )

    root = Path(downloaded_path).resolve()
    files, parquet_files, total_size = scan_files(root)

    info = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "local_dir": str(root),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_files": len(files),
        "num_parquet_files": len(parquet_files),
        "total_size_gib": round(bytes_to_gib(total_size), 3),
        "parquet_files_preview": [str(p.relative_to(root)) for p in sorted(parquet_files)[:10]],
    }

    info_path = root / "_download_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("Download complete.")
    print(f"Files found: {len(files)}")
    print(f"Parquet files found: {len(parquet_files)}")
    print(f"Local size: {bytes_to_gib(total_size):.2f} GiB")
    print(f"Download info written to: {info_path}")


if __name__ == "__main__":
    main()
