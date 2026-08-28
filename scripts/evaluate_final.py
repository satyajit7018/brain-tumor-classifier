#!/usr/bin/env python3
"""Phase 4: Full evaluation suite and Grad-CAM heatmap generator.
Computes confusion matrix, per-class F1, ROC-AUC, and false-negative rate,
saving results to docs/eval_results.json and generating heatmaps in docs/gradcam_examples/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tensorflow as tf
from src.data.dataset import load_dataset_as_numpy, CLASS_NAMES
from src.eval.metrics import evaluate_model
from src.eval.gradcam import make_gradcam_heatmap, overlay_heatmap


def main():
    parser = argparse.ArgumentParser(description="Evaluate final model and generate Grad-CAM heatmaps.")
    parser.add_argument("--model-path", type=str, default="saved_models/best_model.keras", help="Path to saved model")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset")
    parser.add_argument("--output-json", type=str, default="docs/eval_results.json", help="Path for evaluation metrics JSON")
    parser.add_argument("--gradcam-dir", type=str, default="docs/gradcam_examples", help="Directory for Grad-CAM sample images")
    parser.add_argument("--dry-run", action="store_true", help="Generate synthetic evaluations for pipeline verification")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(args.gradcam_dir, exist_ok=True)

    print("=================================================================")
    print("  Brain Tumor Classifier — Phase 4: Full Clinical Evaluation")
    print("=================================================================")

    if args.dry_run:
        print("[INFO] Running in DRY-RUN mode with synthetic model and test samples...")
        from src.models.baseline_cnn import build_baseline_cnn
        model = build_baseline_cnn()
        images = np.random.rand(64, 224, 224, 3).astype(np.float32)
        labels = np.random.randint(0, 4, size=64)
    else:
        if not os.path.exists(args.model_path):
            print(f"[ERROR] Model file not found at {args.model_path}. Train model first (Phase 3).")
            sys.exit(1)

        print(f"[INFO] Loading model from: {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)

        print(f"[INFO] Loading evaluation dataset from: {args.data_dir}")
        images, labels = load_dataset_as_numpy(args.data_dir)

        if len(images) == 0:
            print(f"[ERROR] No evaluation images found in {args.data_dir}.")
            sys.exit(1)

    print(f"[INFO] Running inference on {len(images)} samples...")
    y_pred_probs = model.predict(images, batch_size=32, verbose=1)
    y_preds = np.argmax(y_pred_probs, axis=1)

    print("[INFO] Computing clinical metrics...")
    metrics = evaluate_model(y_true=labels, y_pred_probs=y_pred_probs)

    # Save to docs/eval_results.json
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Summary Results ---")
    print(f"False Negative Rate (Primary Clinical Metric): {metrics['false_negative_rate']:.4%}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"Metrics saved to: {args.output_json}")

    # Generate 8 Grad-CAM examples: 4 correct, 4 incorrect/edge cases
    print(f"\n[INFO] Generating Grad-CAM heatmaps in {args.gradcam_dir}...")
    correct_indices = np.where(labels == y_preds)[0]
    incorrect_indices = np.where(labels != y_preds)[0]

    selected_indices = []
    # Take up to 4 correct samples
    if len(correct_indices) > 0:
        selected_indices.extend(correct_indices[:min(4, len(correct_indices))])
    # Take up to 4 incorrect samples (or more correct if none incorrect)
    if len(incorrect_indices) > 0:
        selected_indices.extend(incorrect_indices[:min(4, len(incorrect_indices))])
    else:
        remaining = 8 - len(selected_indices)
        if len(correct_indices) > 4:
            selected_indices.extend(correct_indices[4:4 + remaining])

    for i, idx in enumerate(selected_indices):
        img_arr = np.expand_dims(images[idx], axis=0)
        true_label = CLASS_NAMES[labels[idx]]
        pred_label = CLASS_NAMES[y_preds[idx]]
        confidence = float(y_pred_probs[idx][y_preds[idx]])
        is_correct = (labels[idx] == y_preds[idx])

        try:
            heatmap, _ = make_gradcam_heatmap(img_arr, model, pred_index=y_preds[idx])
            orig_img = (images[idx] * 255.0).astype(np.uint8)
            overlaid = overlay_heatmap(orig_img, heatmap)

            # Create side-by-side comparison canvas
            h, w, _ = orig_img.shape
            canvas = np.zeros((h + 60, w * 2 + 20, 3), dtype=np.uint8)
            canvas[50:50 + h, :w] = orig_img
            canvas[50:50 + h, w + 20:w * 2 + 20] = overlaid

            # Add labels
            status_text = "CORRECT" if is_correct else "MISCLASSIFIED"
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            header_text = f"[{status_text}] True: {true_label} | Pred: {pred_label} ({confidence:.1%})"
            cv2.putText(canvas, header_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(canvas, "Original Scan", (10, h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, "Grad-CAM Attention Overlay", (w + 30, h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            out_filename = f"gradcam_sample_{i+1}_{status_text.lower()}_true_{true_label}_pred_{pred_label}.png"
            cv2.imwrite(os.path.join(args.gradcam_dir, out_filename), canvas)
            print(f"  [+] Saved {out_filename}")
        except Exception as e:
            print(f"  [-] Failed to generate Grad-CAM for sample {idx}: {e}")

    print("\n=================================================================")
    print(f" [SUCCESS] Phase 4 complete! Heatmaps saved to: {args.gradcam_dir}")
    print("=================================================================")


if __name__ == "__main__":
    main()
