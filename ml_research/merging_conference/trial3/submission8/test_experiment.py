import unittest
import torch
import torch.nn as nn
import os
import json
from experiment import (
    get_resnet18_model,
    compute_spor_loss,
    compute_fisher,
    orthomerge_layers
)

class TestFWSPOR(unittest.TestCase):
    def setUp(self):
        # Create lightweight mock elements
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        ).to(self.device)
        self.base_model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        ).to(self.device)
        
        # Copy weights
        self.base_model[0].weight.data.copy_(self.model[0].weight.data)
        
        # Free base model parameters
        for p in self.base_model.parameters():
            p.requires_grad = False
            
    def test_spor_loss_standard(self):
        # Standard SPOR
        loss = compute_spor_loss(self.model, self.base_model, beta=0.1, gamma=0.0)
        self.assertGreaterEqual(loss.item(), 0.0)
        
    def test_spor_loss_fisher_weighted(self):
        # Mock Fisher
        fisher = {
            '0.weight': torch.ones_like(self.base_model[0].weight) * 2.5
        }
        loss = compute_spor_loss(self.model, self.base_model, beta=0.1, gamma=0.5, fisher=fisher)
        self.assertGreaterEqual(loss.item(), 0.0)
        
    def test_compute_fisher(self):
        # Mock simple dataloader yielding random batches
        class MockLoader:
            def __init__(self, num_batches=2, batch_size=4):
                self.num_batches = num_batches
                self.batch_size = batch_size
                
            def __iter__(self):
                for _ in range(self.num_batches):
                    # resnet expects 3-channel input
                    images = torch.randn(self.batch_size, 3, 128, 128)
                    labels = torch.randint(0, 10, (self.batch_size,))
                    yield images, labels
                    
        # Let's use a very lightweight model for compute_fisher test to avoid slow download or memory load
        mock_model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, 10)
        ).to(self.device)
        
        # We need parameter names containing 'weight' and shape length 4
        # conv weight has shape [4, 3, 3, 3] which matches
        fisher = compute_fisher(mock_model, MockLoader(), self.device, num_samples=8)
        self.assertIn('0.weight', fisher)
        self.assertEqual(fisher['0.weight'].shape, mock_model[0].weight.shape)
        # Max value of normalized Fisher should be 1.0 (or very close depending on floating point precision)
        self.assertAlmostEqual(fisher['0.weight'].max().item(), 1.0, places=5)

    def test_orthomerge_layers(self):
        # Test OrthoMerge calculation
        W_A = torch.randn(8, 27, device=self.device)
        W_B = torch.randn(8, 27, device=self.device)
        W0 = torch.randn(8, 27, device=self.device)
        
        W_merged, avg_res_norm = orthomerge_layers(W_A, W_B, W0)
        self.assertEqual(W_merged.shape, (8, 27))
        self.assertGreater(avg_res_norm, 0.0)

if __name__ == '__main__':
    unittest.main()
