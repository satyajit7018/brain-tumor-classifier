#!/usr/bin/env python3
"""Phase 1 Utility: Automated dataset downloader and organizer.
Handles downloading from Kaggle (if kaggle CLI / kaggle.json is present)
or extracting a local zip archive, organizing into data/raw/{glioma, meningioma, pituitary, no_tumor}.
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CLASS_NAMES, check_class_balance


CLASS_MAP = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "no_tumor",
    "no_tumor": "no_tumor",
}


def organize_extracted_data(source_dir: Path, target_dir: Path):
    """Scan source_dir for Training/ and Testing/ folders and merge into target_dir."""
    print(f"[INFO] Organizing dataset from {source_dir} into {target_dir}...")

    for target_class in CLASS_NAMES:
        (target_dir / target_class).mkdir(parents=True, exist_ok=True)

    copied_count = 0
    # Walk and look for folders matching our classes
    for root, dirs, files in os.walk(source_dir):
        folder_name = os.path.basename(root).lower().replace(" ", "").replace("_", "")
        for source_name, target_class in CLASS_MAP.items():
            if folder_name == source_name.replace("_", ""):
                dest_dir = target_dir / target_class
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        src_path = os.path.join(root, f)
                        dst_path = dest_dir / f"{Path(root).parent.name}_{f}"
                        if not dst_path.exists():
                            shutil.copy2(src_path, dst_path)
                            copied_count += 1

    print(f"[INFO] Organized {copied_count} images into {target_dir}.")
    return copied_count


def download_with_kaggle(download_dir: Path):
    """Attempt download via Kaggle API CLI."""
    print("[INFO] Attempting download via Kaggle CLI...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "masoudnickparvar/brain-tumor-mri-dataset",
        "-p", str(download_dir),
        "--unzip"
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[SUCCESS] Kaggle download completed.")
        return True
    except FileNotFoundError:
        print("[WARNING] Kaggle CLI not found. Install with `pip install kaggle`.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Kaggle download failed: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Brain Tumor MRI dataset.")
    parser.add_argument("--zip-path", type=str, default=None, help="Path to local brain-tumor-mri-dataset.zip")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Target destination for organized images")
    parser.add_argument("--temp-dir", type=str, default="data/temp_extract", help="Temporary extraction directory")
    args = parser.parse_args()

    raw_path = PROJECT_ROOT / args.raw_dir
    temp_path = PROJECT_ROOT / args.temp_dir

    # Check if dataset already exists in data/raw
    balance = check_class_balance(str(raw_path))
    if sum(balance.values()) > 0:
        print(f"[INFO] Data already present in {raw_path}: {balance}")
        print("Dataset is ready.")
        return

    # Check if zip provided or in Downloads
    zip_candidate = None
    if args.zip_path and os.path.exists(args.zip_path):
        zip_candidate = Path(args.zip_path)
    else:
        downloads_zip = Path.home() / "Downloads" / "brain-tumor-mri-dataset.zip"
        archive_zip = Path.home() / "Downloads" / "archive.zip"
        if downloads_zip.exists():
            zip_candidate = downloads_zip
        elif archive_zip.exists():
            zip_candidate = archive_zip

    if zip_candidate:
        print(f"[INFO] Found local archive at {zip_candidate}. Extracting...")
        temp_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_candidate, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
        organize_extracted_data(temp_path, raw_path)
        shutil.rmtree(temp_path, ignore_errors=True)
    else:
        # Try Kaggle CLI
        temp_path.mkdir(parents=True, exist_ok=True)
        success = download_with_kaggle(temp_path)
        if success:
            organize_extracted_data(temp_path, raw_path)
            shutil.rmtree(temp_path, ignore_errors=True)
        else:
            print("\n[INSTRUCTIONS] To acquire the dataset:")
            print("1. Download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
            print(f"2. Run: python scripts/download_dataset.py --zip-path ~/Downloads/archive.zip")

    # Print final balance
    final_balance = check_class_balance(str(raw_path))
    print(f"\n[SUMMARY] Class balance: {final_balance} | Total: {sum(final_balance.values())}")


if __name__ == "__main__":
    main()
