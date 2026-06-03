import unittest
import torch
import torch.nn as nn
import merge_and_eval

class TestMergingAlgorithms(unittest.TestCase):
    def setUp(self):
        # Create simple dummy models (progenitor and two task experts)
        # We use a 2D projection weight (conv), a 2D fc weight, and a 1D batchnorm running mean/weight
        self.progenitor = {
            "conv.weight": torch.ones(2, 2, 3, 3),
            "fc.weight": torch.ones(2, 2),
            "bn.weight": torch.ones(2),
            "bn.running_mean": torch.zeros(2)
        }
        
        self.expert1 = {
            "conv.weight": torch.ones(2, 2, 3, 3) + 0.1,
            "fc.weight": torch.ones(2, 2) + 0.1,
            "bn.weight": torch.ones(2) * 1.1,
            "bn.running_mean": torch.ones(2) * 0.1
        }
        
        self.expert2 = {
            "conv.weight": torch.ones(2, 2, 3, 3) - 0.1,
            "fc.weight": torch.ones(2, 2) - 0.1,
            "bn.weight": torch.ones(2) * 0.9,
            "bn.running_mean": torch.ones(2) * (-0.1)
        }
        self.experts = [self.expert1, self.expert2]
        self.weights = [0.5, 0.5]

    def test_spherical_karcher_mean(self):
        # Create two orthogonal unit vectors
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])
        vectors = [v1, v2]
        weights = [0.5, 0.5]
        
        # Geodesic mean of (1,0) and (0,1) should be at 45 degrees: (1/sqrt(2), 1/sqrt(2))
        expected = torch.tensor([1.0 / 2**0.5, 1.0 / 2**0.5])
        result = merge_and_eval.spherical_karcher_mean(vectors, weights, num_iters=10)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_spherical_karcher_mean_channelwise(self):
        # Create two 2D tensors of shape (2, 2)
        v1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        v2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        vectors = [v1, v2]
        weights = [0.5, 0.5]
        
        # Channel-wise spherical mean computes Karcher mean for each channel (row)
        # Each row is geodesic mean of (1,0) and (0,1), yielding (1/sqrt(2), 1/sqrt(2))
        expected = torch.tensor([[1.0 / 2**0.5, 1.0 / 2**0.5], [1.0 / 2**0.5, 1.0 / 2**0.5]])
        result = merge_and_eval.spherical_karcher_mean_channelwise(vectors, weights, num_iters=10)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_weight_averaging(self):
        # Weight averaging should linearly blend parameters (except fc and num_batches_tracked)
        merged = merge_and_eval.weight_averaging(self.progenitor, self.experts, self.weights)
        expected_conv = torch.ones(2, 2, 3, 3)
        expected_bn_weight = torch.ones(2)
        expected_bn_rm = torch.zeros(2)
        
        torch.testing.assert_close(merged["conv.weight"], expected_conv)
        self.assertNotIn("fc.weight", merged)
        torch.testing.assert_close(merged["bn.weight"], expected_bn_weight)
        torch.testing.assert_close(merged["bn.running_mean"], expected_bn_rm)

    def test_task_arithmetic(self):
        # Task arithmetic adds lambda-scaled task vectors back to progenitor
        # Task vector 1: expert1 - progenitor. Task vector 2: expert2 - progenitor.
        # Average task vector: 0.5 * (0.1) + 0.5 * (-0.1) = 0.0
        # For lambda=0.3, merged = progenitor + 0.3 * (average task vector)
        merged = merge_and_eval.task_arithmetic(self.progenitor, self.experts, lam=0.3)
        
        # Since average task vector is 0, the result should be equal to progenitor (for backbone weights)
        # Note: Task Arithmetic typically skips "fc" and "num_batches_tracked", but processes conv and bn
        self.assertIn("conv.weight", merged)
        torch.testing.assert_close(merged["conv.weight"], self.progenitor["conv.weight"])

    def test_spherical_karcher_merging_selective(self):
        # Selective SKM uses Karcher mean on projection weight (conv.weight)
        # and standard weight averaging on others (bn, fc, biases).
        # Progenitor conv.weight = 1s, Expert1 conv.weight = 1.1s, Expert2 conv.weight = 0.9s
        # Average norm is 1.0. Normalized vectors are all 1s / norm.
        # Geodesic mean should align with normalized sum, scaled by avg norm (which is 1.0).
        # Thus, merged conv.weight should be exactly 1s (all ones).
        merged = merge_and_eval.spherical_karcher_merging(
            self.progenitor, self.experts, weights=self.weights, num_iters=5, selective=True, channelwise=False
        )
        
        expected_conv = torch.ones(2, 2, 3, 3)
        torch.testing.assert_close(merged["conv.weight"], expected_conv)
        # Non-projection parameters should use standard Weight Averaging
        torch.testing.assert_close(merged["bn.weight"], self.progenitor["bn.weight"])

    def test_spherical_karcher_merging_global(self):
        # Global SKM applies Karcher Mean globally to all parameters (except fc and num_batches_tracked)
        merged = merge_and_eval.spherical_karcher_merging(
            self.progenitor, self.experts, weights=self.weights, num_iters=5, selective=False, channelwise=False
        )
        self.assertIn("conv.weight", merged)
        self.assertIn("bn.running_mean", merged)

    def test_ties_merging(self):
        # TIES-merging resolves sign conflicts, sparsifies, and scales
        merged = merge_and_eval.ties_merging(self.progenitor, self.experts, fraction=0.5)
        self.assertIn("conv.weight", merged)
        self.assertEqual(merged["conv.weight"].shape, self.progenitor["conv.weight"].shape)

    def test_dare_merging(self):
        # DARE-merging randomly drops weights and scales
        merged = merge_and_eval.dare_merging(self.progenitor, self.experts, p_drop=0.1)
        self.assertIn("conv.weight", merged)
        self.assertEqual(merged["conv.weight"].shape, self.progenitor["conv.weight"].shape)

    def test_sk_ties_merging(self):
        # Spherical Karcher TIES-Merging
        merged = merge_and_eval.sk_ties_merging(
            self.progenitor, self.experts, fraction=0.5, num_iters=2, channelwise=False
        )
        self.assertIn("conv.weight", merged)
        self.assertEqual(merged["conv.weight"].shape, self.progenitor["conv.weight"].shape)

if __name__ == "__main__":
    unittest.main()
