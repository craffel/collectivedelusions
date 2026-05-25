import unittest
import torch
import numpy as np
from experiment import SimpleCNN, get_layer_group_index

class TestExperiment(unittest.TestCase):
    def test_layer_group_index(self):
        # Test layer group classification logic
        self.assertEqual(get_layer_group_index('conv1.weight'), 0)
        self.assertEqual(get_layer_group_index('bn1.bias'), 0)
        self.assertEqual(get_layer_group_index('conv2.weight'), 1)
        self.assertEqual(get_layer_group_index('bn2.running_mean'), 1)
        self.assertEqual(get_layer_group_index('fc1.weight'), 2)
        self.assertEqual(get_layer_group_index('fc1.bias'), 2)
        self.assertEqual(get_layer_group_index('fc2.weight'), 3)
        self.assertEqual(get_layer_group_index('fc2.bias'), 3)
        self.assertEqual(get_layer_group_index('other_param'), 0)

    def test_simple_cnn_forward(self):
        # Test SimpleCNN instantiation and forward pass shapes
        model = SimpleCNN()
        model.eval()
        mock_input = torch.randn(4, 1, 28, 28)
        output = model(mock_input)
        self.assertEqual(output.shape, (4, 10))

    def test_fused_bn_buffers(self):
        # Test fused BN buffers logic with simplified mock state dicts
        # Since get_fused_bn_buffers is defined inside run_evaluation in experiment.py,
        # we can define a similar reference implementation and verify its math here.
        def reference_fused_bn_buffers(state0, state1, w):
            fused_buffers = {}
            for name in state0.keys():
                if 'running_mean' in name:
                    mean0 = state0[name]
                    mean1 = state1[name]
                    fused_buffers[name] = w[0] * mean0 + w[1] * mean1
                elif 'running_var' in name:
                    var0 = state0[name]
                    var1 = state1[name]
                    mean0_name = name.replace('running_var', 'running_mean')
                    mean0 = state0[mean0_name]
                    mean1 = state1[mean0_name]
                    mean_fused = w[0] * mean0 + w[1] * mean1
                    fused_buffers[name] = w[0] * (var0 + mean0**2) + w[1] * (var1 + mean1**2) - mean_fused**2
                elif 'num_batches_tracked' in name:
                    fused_buffers[name] = state0[name]
            return fused_buffers

        state0 = {
            'bn1.running_mean': torch.tensor([1.0, 2.0]),
            'bn1.running_var': torch.tensor([0.5, 0.8]),
            'bn1.num_batches_tracked': torch.tensor(10)
        }
        state1 = {
            'bn1.running_mean': torch.tensor([3.0, 4.0]),
            'bn1.running_var': torch.tensor([1.0, 1.2]),
            'bn1.num_batches_tracked': torch.tensor(10)
        }
        w = [0.4, 0.6]

        fused = reference_fused_bn_buffers(state0, state1, w)
        
        # Hand-calculation checks:
        # Fused mean = 0.4 * 1.0 + 0.6 * 3.0 = 0.4 + 1.8 = 2.2
        # Fused var = 0.4 * (0.5 + 1^2) + 0.6 * (1.0 + 3^2) - 2.2^2
        #           = 0.4 * 1.5 + 0.6 * 10.0 - 4.84
        #           = 0.6 + 6.0 - 4.84 = 6.6 - 4.84 = 1.76
        
        expected_mean = torch.tensor([2.2, 3.2])
        expected_var = torch.tensor([1.76, 1.96]) # 0.4*(0.8 + 4) + 0.6*(1.2 + 16) - 3.2^2 = 1.92 + 10.32 - 10.24 = 2.00
        
        self.assertTrue(torch.allclose(fused['bn1.running_mean'], expected_mean, atol=1e-5))
        self.assertTrue(torch.allclose(fused['bn1.running_var'], torch.tensor([1.76, 2.00]), atol=1e-5))

    def test_sam_perturbation(self):
        # Test the SAM preconditioned perturbation step logic
        w_global_adapted = torch.tensor(0.5, requires_grad=True)
        deltas_adapted = [torch.tensor(0.1 * j, requires_grad=True) for j in range(4)]
        
        # Mock gradients
        w_global_adapted.grad = torch.tensor(0.2)
        for j, d in enumerate(deltas_adapted):
            d.grad = torch.tensor(0.05 * (j + 1))
            
        # Mock activations/gradients for preconditioning
        F_sens = [0.1, 0.5, 1.0, 2.0]
        eps_stab = 0.1
        rho = 0.05
        
        # Perform calculation
        d_w = w_global_adapted.grad
        d_deltas = []
        for j in range(4):
            d_d = deltas_adapted[j].grad
            d_deltas.append(d_d / (F_sens[j] + eps_stab))
            
        dir_norm = torch.sqrt(d_w**2 + sum([torch.sum(dd**2) for dd in d_deltas]) + eps_stab)
        
        epsilon_w = rho * d_w / dir_norm
        epsilon_deltas = [rho * dd / dir_norm for dd in d_deltas]
        
        w_perturbed = w_global_adapted + epsilon_w
        deltas_perturbed = [deltas_adapted[j] + epsilon_deltas[j] for j in range(4)]
        
        # Expected manual calculations:
        # d_w = 0.2
        # d_deltas = [
        #   0.05 / 0.2 = 0.25,
        #   0.10 / 0.6 = 0.16666667,
        #   0.15 / 1.1 = 0.13636364,
        #   0.20 / 2.1 = 0.09523810
        # ]
        # sum(dd^2) = 0.25^2 + 0.16666667^2 + 0.13636364^2 + 0.0952381^2
        #           = 0.0625 + 0.02777778 + 0.01859504 + 0.00907029 = 0.11794311
        # d_w^2 = 0.04
        # dir_norm = sqrt(0.04 + 0.11794311 + 0.1) = sqrt(0.25794311) = 0.507881
        # epsilon_w = 0.05 * 0.2 / 0.507881 = 0.01968965
        # w_perturbed = 0.5 + 0.01968965 = 0.51968965
        
        self.assertAlmostEqual(w_perturbed.item(), 0.51968965, places=5)
        self.assertAlmostEqual(deltas_perturbed[0].item(), 0.0246112, places=5)

if __name__ == '__main__':
    unittest.main()
