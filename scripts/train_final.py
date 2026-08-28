#!/usr/bin/env python3
"""Phase 3: Train the final deployable champion model and save weights
to saved_models/best_model.keras.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from src.data.dataset import load_datasets, check_class_balance, compute_class_weights, CLASS_NAMES
from src.data.augmentation import apply_augmentation
from src.train.train import MODEL_BUILDERS


def main():
    parser = argparse.ArgumentParser(description="Train the final champion model for deployment.")
    parser.add_argument("--model", type=str, default="resnet50", choices=list(MODEL_BUILDERS.keys()),
                        help="Model architecture to train (resnet50, efficientnet_b0, baseline_cnn)")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set split ratio")
    parser.add_argument("--output-model", type=str, default="saved_models/best_model.keras", help="Path to save best model")
    parser.add_argument("--use-augmentation", action="store_true", default=True, help="Apply data augmentation during training")
    parser.add_argument("--use-class-weights", action="store_true", default=True, help="Apply inverse-frequency class weights")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch on synthetic data for verification")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model) or "saved_models", exist_ok=True)

    print("=================================================================")
    print(f"  Brain Tumor Classifier — Phase 3: Train Final Model ({args.model.upper()})")
    print("=================================================================")

    if args.dry_run:
        print("[INFO] Running in DRY-RUN mode with synthetic dataset...")
        import numpy as np
        x_train = np.random.rand(64, 224, 224, 3).astype(np.float32)
        y_train = np.random.randint(0, 4, size=64)
        x_val = np.random.rand(32, 224, 224, 3).astype(np.float32)
        y_val = np.random.randint(0, 4, size=32)

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)
        class_weights = None
        epochs = 1
    else:
        # Check dataset existence
        balance = check_class_balance(args.data_dir)
        total_images = sum(balance.values())
        if total_images == 0:
            print(f"[ERROR] No images found in {args.data_dir}. Check dataset setup in Phase 1.")
            sys.exit(1)

        print(f"[INFO] Dataset balance in {args.data_dir}: {balance} (Total: {total_images})")

        train_ds, val_ds = load_datasets(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
        )

        if args.use_augmentation:
            print("[INFO] Enabling training data augmentation...")
            train_ds = apply_augmentation(train_ds)

        class_weights = None
        if args.use_class_weights:
            class_weights = compute_class_weights(balance)
            print(f"[INFO] Computed class weights: {class_weights}")

        epochs = args.epochs

    # Build model
    print(f"[INFO] Building {args.model} architecture...")
    model = MODEL_BUILDERS[args.model]()

    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output_model,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n[INFO] Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Save final model if not saved yet by checkpoint
    if not os.path.exists(args.output_model):
        model.save(args.output_model)

    # Save training metadata
    meta_path = os.path.splitext(args.output_model)[0] + "_history.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model": args.model,
            "epochs_trained": len(history.history.get("loss", [])),
            "final_val_loss": float(history.history.get("val_loss", [-1])[-1]),
            "final_val_accuracy": float(history.history.get("val_accuracy", [-1])[-1]),
        }, f, indent=2)

    print("\n=================================================================")
    print(f" [SUCCESS] Final model successfully trained and saved to: {args.output_model}")
    print("=================================================================")


if __name__ == "__main__":
    main()
