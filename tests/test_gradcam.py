"""Unit tests for Grad-CAM layer resolution and heatmap generation."""

import unittest
import numpy as np

from src.eval.gradcam import overlay_heatmap


class TestGradCAMModule(unittest.TestCase):
    def test_overlay_heatmap_shapes(self):
        original_img = np.random.rand(224, 224, 3).astype(np.float32)
        heatmap = np.random.rand(7, 7).astype(np.float32)

        overlaid = overlay_heatmap(original_img, heatmap, alpha=0.4)
        self.assertEqual(overlaid.shape, (224, 224, 3))
        self.assertEqual(overlaid.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
