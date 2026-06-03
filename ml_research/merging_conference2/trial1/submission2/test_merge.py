import unittest
import torch
import torch.nn as nn
from merge import merge_models, bures_wasserstein_barycenter, sym_matrix_power

class TestModelMerging(unittest.TestCase):
    def setUp(self):
        # Create dummy base and task-specific state dicts
        torch.manual_seed(42)
        
        self.base_state = {
            'layer1.weight': torch.randn(8, 8),
            'layer1.bias': torch.randn(8),
            'layer2.weight': torch.randn(16, 8),
            'layer2.bias': torch.randn(16)
        }
        
        # Two fine-tuned models
        self.task_states = [
            {
                'layer1.weight': self.base_state['layer1.weight'] + 0.1 * torch.randn(8, 8),
                'layer1.bias': self.base_state['layer1.bias'] + 0.1 * torch.randn(8),
                'layer2.weight': self.base_state['layer2.weight'] + 0.1 * torch.randn(16, 8),
                'layer2.bias': self.base_state['layer2.bias'] + 0.1 * torch.randn(16)
            },
            {
                'layer1.weight': self.base_state['layer1.weight'] + 0.2 * torch.randn(8, 8),
                'layer1.bias': self.base_state['layer1.bias'] + 0.2 * torch.randn(8),
                'layer2.weight': self.base_state['layer2.weight'] + 0.2 * torch.randn(16, 8),
                'layer2.bias': self.base_state['layer2.bias'] + 0.2 * torch.randn(16)
            }
        ]
        
        self.weights = [0.5, 0.5]

    def test_task_arithmetic(self):
        merged = merge_models(self.base_state, self.task_states, self.weights, 'task_arithmetic')
        
        # Verify shape
        self.assertEqual(merged['layer1.weight'].shape, (8, 8))
        self.assertEqual(merged['layer2.weight'].shape, (16, 8))
        
        # Verify standard linear arithmetic
        expected_layer1_bias = self.base_state['layer1.bias'] + 0.5 * (self.task_states[0]['layer1.bias'] - self.base_state['layer1.bias']) + 0.5 * (self.task_states[1]['layer1.bias'] - self.base_state['layer1.bias'])
        torch.testing.assert_close(merged['layer1.bias'], expected_layer1_bias)

    def test_isotropic(self):
        merged = merge_models(self.base_state, self.task_states, self.weights, 'isotropic')
        self.assertEqual(merged['layer1.weight'].shape, (8, 8))
        self.assertEqual(merged['layer2.weight'].shape, (16, 8))

    def test_wsa_unnormalized(self):
        merged = merge_models(self.base_state, self.task_states, self.weights, 'wsa', normalize_barycenter=False)
        self.assertEqual(merged['layer1.weight'].shape, (8, 8))
        self.assertEqual(merged['layer2.weight'].shape, (16, 8))

    def test_wsa_normalized(self):
        merged = merge_models(self.base_state, self.task_states, self.weights, 'wsa', normalize_barycenter=True)
        self.assertEqual(merged['layer1.weight'].shape, (8, 8))
        self.assertEqual(merged['layer2.weight'].shape, (16, 8))

    def test_bures_wasserstein_barycenter_homogeneity(self):
        # Generate positive definite matrices
        A = torch.randn(4, 4)
        cov1 = A @ A.T + 0.1 * torch.eye(4)
        B = torch.randn(4, 4)
        cov2 = B @ B.T + 0.1 * torch.eye(4)
        
        covs = [cov1, cov2]
        weights = [0.5, 0.5]
        
        # Barycenter under normal mode
        bary1 = bures_wasserstein_barycenter(covs, weights, normalize=False)
        
        # Barycenter under normalized mode (homogeneous scaling)
        bary2 = bures_wasserstein_barycenter(covs, weights, normalize=True)
        
        # Since they are both valid barycenters (even if convergence speeds vary), they should be positive definite
        eig1, _ = torch.linalg.eigh(bary1)
        eig2, _ = torch.linalg.eigh(bary2)
        
        self.assertTrue(torch.all(eig1 > 0))
        self.assertTrue(torch.all(eig2 > 0))

if __name__ == '__main__':
    unittest.main()
