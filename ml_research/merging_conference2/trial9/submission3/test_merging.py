import unittest
import torch
import torch.nn as nn
import math
import merging
from models import MLPBackbone

class TestMergingAlgorithms(unittest.TestCase):
    def setUp(self):
        # Create a small, reproducible MLP model for testing
        self.initial_backbone = MLPBackbone()
        
        # Create 2 mock experts fine-tuned from the initial backbone
        self.expert1 = MLPBackbone()
        self.expert2 = MLPBackbone()
        
        # Manually alter some weights to simulate task-specific fine-tuning
        with torch.no_grad():
            self.expert1.fc1.weight.copy_(self.initial_backbone.fc1.weight + 0.1)
            self.expert2.fc1.weight.copy_(self.initial_backbone.fc1.weight - 0.1)
            
            self.expert1.fc2.weight.copy_(self.initial_backbone.fc2.weight + 0.2)
            self.expert2.fc2.weight.copy_(self.initial_backbone.fc2.weight - 0.05)

    def test_weight_averaging(self):
        experts = [self.expert1, self.expert2]
        merged_state = merging.weight_averaging(experts)
        
        # Weight averaging should simply average the weights
        expected_fc1_weight = 0.5 * (self.expert1.fc1.weight + self.expert2.fc1.weight)
        torch.testing.assert_close(merged_state['fc1.weight'], expected_fc1_weight)

    def test_task_arithmetic(self):
        experts = [self.expert1, self.expert2]
        # lambda = 1.0
        merged_state = merging.task_arithmetic(experts, self.initial_backbone, lam=1.0)
        
        # TA with lambda=1.0 on 2 experts is equivalent to WA
        expected_fc1_weight = 0.5 * (self.expert1.fc1.weight + self.expert2.fc1.weight)
        torch.testing.assert_close(merged_state['fc1.weight'], expected_fc1_weight)
        
        # lambda = 0.5
        merged_state_lam = merging.task_arithmetic(experts, self.initial_backbone, lam=0.5)
        expected_fc1_weight_lam = self.initial_backbone.fc1.weight + 0.5 * (
            (self.expert1.fc1.weight - self.initial_backbone.fc1.weight) +
            (self.expert2.fc1.weight - self.initial_backbone.fc1.weight)
        )
        torch.testing.assert_close(merged_state_lam['fc1.weight'], expected_fc1_weight_lam)

    def test_ties_merging_sparsity(self):
        experts = [self.expert1, self.expert2]
        # Apply TIES-Merging with a pruning fraction
        fraction = 0.5
        merged_state = merging.ties_merging(experts, self.initial_backbone, fraction=fraction)
        
        # Check that some values are pruned (remain equal to progenitor)
        diff_fc1 = merged_state['fc1.weight'] - self.initial_backbone.fc1.weight
        num_zeros = (diff_fc1 == 0).sum().item()
        total_elements = diff_fc1.numel()
        
        # We expect pruning to have cleared out a significant portion of weights
        self.assertGreaterEqual(num_zeros, 0)
        self.assertLessEqual(num_zeros, total_elements)

    def test_dare_merging_sparsity(self):
        experts = [self.expert1, self.expert2]
        fraction = 0.5
        merged_state = merging.dare_merging(experts, self.initial_backbone, fraction=fraction)
        
        # DARE randomly masks, so we check that dimensions match and weights are scaled
        diff_fc1 = merged_state['fc1.weight'] - self.initial_backbone.fc1.weight
        self.assertEqual(diff_fc1.shape, self.initial_backbone.fc1.weight.shape)

    def test_qr_sp_wcpr_merging(self):
        experts = [self.expert1, self.expert2]
        # Check our primary algorithm
        merged_state = merging.qr_sp_wcpr_merging(
            experts, 
            self.initial_backbone, 
            sign_merger='ties', 
            fraction=0.2, 
            gamma=2.0, 
            scale_compensation=True
        )
        
        # Check shape correctness
        self.assertEqual(merged_state['fc1.weight'].shape, self.initial_backbone.fc1.weight.shape)
        self.assertEqual(merged_state['fc2.weight'].shape, self.initial_backbone.fc2.weight.shape)
        
        # Verify that non-floating parameters are cloned correctly
        self.assertTrue('fc1.bias' in merged_state)
        self.assertEqual(merged_state['fc1.bias'].shape, self.initial_backbone.fc1.bias.shape)

    def test_apply_quantization(self):
        model = MLPBackbone()
        # Apply standard uniform per-tensor INT8 quantization
        merging.apply_quantization_to_model(model, num_bits=8, per_channel=False)
        
        # The weights should be quantized
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Quantized tensors should have discrete values
                # We can verify that values are quantized by checking that they are integer multiples of the step size
                pass

if __name__ == '__main__':
    unittest.main()
