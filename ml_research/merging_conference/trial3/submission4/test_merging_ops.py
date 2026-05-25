import torch
import torch.nn as nn
import unittest
import copy

# Import functions to test
from merge import task_arithmetic_merge, ortho_merge_layer
from train import get_spor_loss

class TestMergingAndRegularization(unittest.TestCase):
    def setUp(self):
        # Create a simple toy model for testing
        self.base_model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        )
        self.model_A = copy.deepcopy(self.base_model)
        self.model_B = copy.deepcopy(self.base_model)
        
        # Modify weights
        with torch.no_grad():
            self.model_A[0].weight.add_(0.1)
            self.model_B[0].weight.sub_(0.1)

    def test_task_arithmetic_merge(self):
        # Merge with lambda = 0.5
        model_merged = task_arithmetic_merge(self.model_A, self.model_B, self.base_model, lam=0.5)
        
        # Expected is base_model + 0.5 * (tau_A + tau_B)
        # since tau_A = 0.1 and tau_B = -0.1, tau_A + tau_B = 0, so merged should equal base_model
        for p_m, p_0 in zip(model_merged.parameters(), self.base_model.parameters()):
            self.assertTrue(torch.allclose(p_m, p_0, atol=1e-6))

    def test_ortho_merge_layer(self):
        W_A = self.model_A[0].weight
        W_B = self.model_B[0].weight
        W_0 = self.base_model[0].weight
        
        W_merged = ortho_merge_layer(W_A, W_B, W_0)
        self.assertEqual(W_merged.shape, W_0.shape)
        # Check that it produces finite, non-NaN values
        self.assertTrue(torch.isfinite(W_merged).all())

    def test_get_spor_loss_standard(self):
        # Test standard SPOR loss calculation (no running_fisher, no fg_mode)
        loss = get_spor_loss(self.model_A, self.base_model, beta=0.1, running_fisher=None, fg_mode=None)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_get_spor_loss_fg_direct(self):
        # Setup running fisher with dummy sensitivities (e.g. 1.0 for first 4 filters, 2.0 for last 4)
        running_fisher = {
            "0.weight": torch.cat([torch.ones(4), torch.ones(4) * 2.0])
        }
        
        loss = get_spor_loss(self.model_A, self.base_model, beta=0.1, running_fisher=running_fisher, fg_mode="direct")
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_get_spor_loss_fg_inverse(self):
        running_fisher = {
            "0.weight": torch.cat([torch.ones(4), torch.ones(4) * 2.0])
        }
        
        loss = get_spor_loss(self.model_A, self.base_model, beta=0.1, running_fisher=running_fisher, fg_mode="inverse")
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

if __name__ == "__main__":
    unittest.main()
