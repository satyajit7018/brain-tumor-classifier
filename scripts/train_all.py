#!/usr/bin/env python3
"""Phase 2: Train all three models (baseline_cnn, resnet50, efficientnet_b0)
with k-fold cross-validation and save benchmark metrics to docs/kfold_results.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.data.dataset import load_dataset_as_numpy, CLASS_NAMES
from src.train.train import run_kfold, MODEL_BUILDERS


def main():
    parser = argparse.ArgumentParser(description="Run k-fold cross-validation on all 3 architectures.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per fold")
    parser.add_argument("--output-json", type=str, default="docs/kfold_results.json", help="Output path for benchmark metrics")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per class (for faster testing)")
    parser.add_argument("--dry-run", action="store_true", help="Generate synthetic test validation without real dataset")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    print("=================================================================")
    print("  Brain Tumor Classifier — Phase 2: K-Fold Architecture Benchmark")
    print("=================================================================")

    if args.dry_run:
        print("[INFO] Running in DRY-RUN mode with synthetic dataset tensors...")
        images = np.random.rand(100, 224, 224, 3).astype(np.float32)
        labels = np.random.randint(0, 4, size=100)
        epochs = 1
        k_folds = 2
    else:
        print(f"[INFO] Loading dataset from: {args.data_dir}")
        images, labels = load_dataset_as_numpy(args.data_dir, max_samples_per_class=args.max_samples)

        if len(images) == 0:
            print(f"[ERROR] No images found in {args.data_dir} across {CLASS_NAMES}.")
            print("Please populate data/raw/ or run with --dry-run for pipeline testing.")
            sys.exit(1)

        print(f"[INFO] Successfully loaded {len(images)} images across {len(CLASS_NAMES)} classes.")
        epochs = args.epochs
        k_folds = args.k_folds

    results_summary = {
        "benchmark_date": str(np.datetime64("now")),
        "k_folds": k_folds,
        "epochs_per_fold": epochs,
        "total_samples": int(len(images)),
        "models": {},
    }

    model_names = ["baseline_cnn", "resnet50", "efficientnet_b0"]

    for model_name in model_names:
        print(f"\n------------------------------------------------------------")
        print(f"  Benchmarking Architecture: {model_name.upper()} ({k_folds}-fold CV)")
        print(f"------------------------------------------------------------")

        try:
            model_results = run_kfold(
                model_name=model_name,
                images=images,
                labels=labels,
                k=k_folds,
                epochs=epochs,
            )
            results_summary["models"][model_name] = model_results
            print(f"\n[DONE] {model_name} Mean Accuracy: {model_results['mean_accuracy']:.4f} (+/- {model_results['std_accuracy']:.4f})")
        except Exception as e:
            print(f"[ERROR] Failed training {model_name}: {e}")
            results_summary["models"][model_name] = {"error": str(e)}

    # Save to docs/kfold_results.json
    with open(args.output_json, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n=================================================================")
    print(f" [SUCCESS] Benchmark completed! Results written to: {args.output_json}")
    print("=================================================================")


if __name__ == "__main__":
    main()
