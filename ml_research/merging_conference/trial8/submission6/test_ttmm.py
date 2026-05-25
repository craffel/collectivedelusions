import unittest
import torch
import torch.nn as nn
import numpy as np

from evaluate_ttmm import (
    get_resnet18_1channel,
    project_simplex,
    get_merged_state_dict
)

class TestTTMM(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

    def test_get_resnet18_1channel(self):
        """Verify the modified ResNet-18 model architecture structure."""
        model = get_resnet18_1channel()
        self.assertIsInstance(model, nn.Module)
        
        # Check input convolutional layer channel surgery
        self.assertEqual(model.conv1.in_channels, 1)
        self.assertEqual(model.conv1.out_channels, 64)
        self.assertEqual(model.conv1.kernel_size, (7, 7))
        
        # Check classification head dimensions
        self.assertEqual(model.fc.in_features, 512)
        self.assertEqual(model.fc.out_features, 10)

    def test_project_simplex(self):
        """Verify that different input tensors are projected onto a valid probability simplex."""
        # Test 2D projection
        v2 = torch.tensor([[2.0, -1.0], [0.5, 0.5], [-10.0, 50.0]])
        y2 = project_simplex(v2)
        
        # Elements must be between 0 and 1
        self.assertTrue(torch.all(y2 >= 0.0))
        self.assertTrue(torch.all(y2 <= 1.0))
        # Sum across last dimension must be 1.0 (with small tolerance)
        sums2 = y2.sum(dim=-1)
        torch.testing.assert_close(sums2, torch.ones_like(sums2))

        # Test higher-dimensional projection
        v5 = torch.randn(10, 5)
        y5 = project_simplex(v5)
        
        self.assertTrue(torch.all(y5 >= -1e-6)) # allow small numerical precision tolerance
        self.assertTrue(torch.all(y5 <= 1.0 + 1e-6))
        sums5 = y5.sum(dim=-1)
        torch.testing.assert_close(sums5, torch.ones_like(sums5))

    def test_soft_bn_buffer_fusion(self):
        """Verify that Soft BN Buffer Fusion correctly performs moment-matching on Mixture of Gaussians."""
        # Create dummy expert state dicts containing running mean and var of a BN layer
        # For simplicity, we create a dict containing one running_mean and one running_var
        sd0 = {
            "layer.running_mean": torch.tensor([1.0, 2.0]),
            "layer.running_var": torch.tensor([0.5, 1.0]),
            "layer.num_batches_tracked": torch.tensor(10)
        }
        sd1 = {
            "layer.running_mean": torch.tensor([3.0, -1.0]),
            "layer.running_var": torch.tensor([0.8, 0.4]),
            "layer.num_batches_tracked": torch.tensor(15)
        }
        
        # Coefficients / weights
        w_bn = torch.tensor([0.4, 0.6])
        coefs = {"layer": torch.tensor([0.5, 0.5])}
        
        merged_sd = get_merged_state_dict(sd0, sd1, coefs, w_bn=w_bn)
        
        # Expected results according to the MoG moment-matching formulas:
        # mu_fused = w0 * mu0 + w1 * mu1
        # var_fused = w0 * (var0 + (mu0 - mu_fused)**2) + w1 * (var1 + (mu1 - mu_fused)**2)
        
        mu0, mu1 = sd0["layer.running_mean"], sd1["layer.running_mean"]
        var0, var1 = sd0["layer.running_var"], sd1["layer.running_var"]
        
        expected_mu = w_bn[0] * mu0 + w_bn[1] * mu1
        expected_var = w_bn[0] * (var0 + (mu0 - expected_mu)**2) + w_bn[1] * (var1 + (mu1 - expected_mu)**2)
        
        torch.testing.assert_close(merged_sd["layer.running_mean"], expected_mu)
        torch.testing.assert_close(merged_sd["layer.running_var"], expected_var)
        self.assertEqual(merged_sd["layer.num_batches_tracked"], sd0["layer.num_batches_tracked"])

    def test_parameter_merging(self):
        """Verify that weights and biases are merged correctly according to layer-specific coefficients."""
        sd0 = {
            "conv1.weight": torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
            "fc.bias": torch.tensor([0.5, -0.5])
        }
        sd1 = {
            "conv1.weight": torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]]),
            "fc.bias": torch.tensor([1.5, 0.5])
        }
        
        # Layer-specific coefficients
        coefs = {
            "conv1": torch.tensor([0.7, 0.3]),
            "fc": torch.tensor([0.2, 0.8])
        }
        
        merged_sd = get_merged_state_dict(sd0, sd1, coefs)
        
        # Conv layer expected weight
        expected_conv_weight = 0.7 * sd0["conv1.weight"] + 0.3 * sd1["conv1.weight"]
        # FC bias expected weight
        expected_fc_bias = 0.2 * sd0["fc.bias"] + 0.8 * sd1["fc.bias"]
        
        torch.testing.assert_close(merged_sd["conv1.weight"], expected_conv_weight)
        torch.testing.assert_close(merged_sd["fc.bias"], expected_fc_bias)

if __name__ == "__main__":
    unittest.main()
