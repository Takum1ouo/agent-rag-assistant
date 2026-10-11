"""
One-off helper script: download the embedding model files to a local folder.

Background: in some network environments the download logic built into the
`huggingface_hub` library fails (it first sends a HEAD request to validate file
metadata and raises FileMetadataError), while a plain GET with requests/curl for the
same files works fine. This script bypasses that logic and downloads the required
files directly with requests into a local folder; afterwards ingest.py and query.py
load the model from that folder and do not go online. It downloads from the
hf-mirror.com mirror by default; to use huggingface.co instead, change MIRROR_BASE below.

Usage (after `conda activate agent-rag`, from the project root), once:
    python src/download_model.py
Afterwards models/paraphrase-multilingual-MiniLM-L12-v2/ contains the model files and
this does not need to be run again.
"""

from pathlib import Path

import requests

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIRROR_BASE = f"https://hf-mirror.com/{REPO}/resolve/main"

FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "1_Pooling/config.json",
    "model.safetensors",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"


def download_file(relative_path: str):
    dest = TARGET_DIR / relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR_BASE}/{relative_path}"

    print(f"Downloading {relative_path} ...", end=" ", flush=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                total += len(chunk)
    print(f"done ({total / 1024 / 1024:.1f} MB)")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for relative_path in FILES:
        download_file(relative_path)
    print(f"\nAll files downloaded. Model saved to: {TARGET_DIR}")


if __name__ == "__main__":
    main()
