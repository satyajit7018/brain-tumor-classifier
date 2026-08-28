#!/usr/bin/env python3
"""Utility to generate synthetic MRI scan samples for pipeline testing.
Generates simulated brain MRI images with circular brain masks and tumor-like intensity features.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.data.dataset import CLASS_NAMES


def generate_synthetic_mri(class_name: str, img_size=(224, 224)) -> np.ndarray:
    """Generate a stylized synthetic brain MRI image."""
    h, w = img_size
    img = np.zeros((h, w), dtype=np.uint8)

    # 1. Background noise
    noise = np.random.normal(15, 5, (h, w)).astype(np.uint8)
    img = cv2.add(img, noise)

    # 2. Draw skull & brain ellipse
    center = (w // 2, h // 2)
    axes = (int(w * 0.38), int(h * 0.44))
    cv2.ellipse(img, center, axes, 0, 0, 360, 160, -1)

    # 3. Brain tissue texture (inner cortex/ventricles)
    brain_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(brain_mask, center, (axes[0] - 10, axes[1] - 10), 0, 0, 360, 255, -1)

    tissue_pattern = np.random.normal(130, 25, (h, w)).astype(np.uint8)
    img = np.where(brain_mask == 255, cv2.addWeighted(img, 0.4, tissue_pattern, 0.6, 0), img)

    # Ventricles in center
    cv2.ellipse(img, (center[0] - 15, center[1]), (8, 28), 10, 0, 360, 40, -1)
    cv2.ellipse(img, (center[0] + 15, center[1]), (8, 28), -10, 0, 360, 40, -1)

    # 4. Add tumor-specific characteristics
    if class_name == "glioma":
        # Infiltrative hyperintense mass with irregular boundary
        t_center = (center[0] + np.random.randint(-40, 40), center[1] + np.random.randint(-40, 40))
        cv2.circle(img, t_center, np.random.randint(22, 35), 235, -1)
        cv2.circle(img, t_center, np.random.randint(10, 18), 255, -1)
    elif class_name == "meningioma":
        # Well-demarcated dural-based extra-axial mass
        t_center = (center[0] + int(axes[0] * 0.7), center[1] + np.random.randint(-30, 30))
        cv2.ellipse(img, t_center, (20, 30), 20, 0, 360, 245, -1)
    elif class_name == "pituitary":
        # Sellar/suprasellar mass near base of skull
        t_center = (center[0], center[1] + int(axes[1] * 0.55))
        cv2.ellipse(img, t_center, (18, 14), 0, 0, 360, 250, -1)
    # no_tumor: normal brain without focal mass

    # Smooth & convert to 3-channel RGB
    img_blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
    img_rgb = cv2.cvtColor(img_blurred, cv2.COLOR_GRAY2RGB)
    return img_rgb


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic brain MRI images for rapid testing.")
    parser.add_argument("--samples-per-class", type=int, default=15, help="Number of samples to generate per class")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target output directory")
    args = parser.parse_args()

    target_dir = PROJECT_ROOT / args.output_dir

    print("=================================================================")
    print(f"  Generating {args.samples_per_class} Synthetic MRI Images Per Class in {target_dir}")
    print("=================================================================")

    total_created = 0
    for cls in CLASS_NAMES:
        cls_dir = target_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.samples_per_class):
            img = generate_synthetic_mri(cls)
            out_file = cls_dir / f"synthetic_{cls}_{i+1:03d}.jpg"
            cv2.imwrite(str(out_file), img)
            total_created += 1

    print(f"\n[SUCCESS] Successfully generated {total_created} synthetic MRI scans across {len(CLASS_NAMES)} classes.")
    print("You can now test k-fold training, final training, API, and Grad-CAM directly!")


if __name__ == "__main__":
    main()
