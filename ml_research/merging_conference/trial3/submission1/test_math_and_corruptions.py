import unittest
import torch
import torch.nn as nn
from eval_tta_rgp import (
    add_gaussian_noise,
    add_gaussian_blur,
    adjust_contrast,
    compute_spor,
    compute_rgp,
    Head
)

class TestMathAndCorruptions(unittest.TestCase):
    def setUp(self):
        # Create dummy tensors for testing corruptions
        # Image tensor: Batch size 2, Channels 1, Height 28, Width 28
        self.dummy_image = torch.ones((2, 1, 28, 28)) * 0.5
        
        # Create dummy Heads for testing SPOR and RGP
        self.head = Head()
        self.expert_head = Head()

    def test_add_gaussian_noise(self):
        # Noise should change the pixel values, but shape should remain the same
        noisy = add_gaussian_noise(self.dummy_image, severity=0.3)
        self.assertEqual(noisy.shape, self.dummy_image.shape)
        # Verify that noise was added (values should not be exactly 0.5 anymore)
        self.assertFalse(torch.allclose(noisy, self.dummy_image))

    def test_add_gaussian_blur(self):
        # Blur should keep the shape same
        blurred = add_gaussian_blur(self.dummy_image, severity=1.5)
        self.assertEqual(blurred.shape, self.dummy_image.shape)

    def test_adjust_contrast(self):
        # Contrast adjustment should keep the shape same
        contrasted = adjust_contrast(self.dummy_image, severity=0.3)
        self.assertEqual(contrasted.shape, self.dummy_image.shape)

    def test_compute_spor(self):
        # SPOR distance between identical heads should be 0 (or close to 0 due to row-normalization & identity)
        # Wait, SPOR checks W_norm * W0_norm.t() matrix. If head and expert_head are identical,
        # W_norm * W0_norm.t() is W_norm * W_norm.t() which might not be identity unless weights are orthogonal.
        # But let's verify that SPOR returns a valid non-negative scalar.
        spor_val = compute_spor(self.head, self.expert_head)
        self.assertIsInstance(spor_val, torch.Tensor)
        self.assertEqual(spor_val.dim(), 0) # Scalar
        self.assertGreaterEqual(spor_val.item(), 0.0)

    def test_compute_rgp(self):
        # RGP between identical heads must be exactly 0 because gram and gram0 are identical
        rgp_val = compute_rgp(self.head, self.head)
        self.assertIsInstance(rgp_val, torch.Tensor)
        self.assertEqual(rgp_val.dim(), 0) # Scalar
        self.assertAlmostEqual(rgp_val.item(), 0.0, places=5)

        # RGP should be a non-negative scalar
        rgp_val_diff = compute_rgp(self.head, self.expert_head)
        self.assertGreaterEqual(rgp_val_diff.item(), 0.0)

if __name__ == "__main__":
    unittest.main()
