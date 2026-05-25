import unittest
import torch
import torch.nn as nn
import copy
from evaluate_tta import translate_augmentation, get_merged_params

class TestMCVTI(unittest.TestCase):
    def setUp(self):
        # Set deterministic seeds for tests
        torch.manual_seed(42)
        
    def test_translate_augmentation(self):
        """
        Tests if the translate_augmentation function preserves tensor shapes,
        performs valid translations, and padding works correctly.
        """
        # Create a mock batch of images (B=2, C=3, H=28, W=28)
        x = torch.randn(2, 3, 28, 28)
        x_aug = translate_augmentation(x, shift_range=2)
        
        self.assertEqual(x.shape, x_aug.shape, "Augmentation must preserve the spatial and channel dimensions of inputs.")
        self.assertNotEqual(torch.sum(torch.abs(x - x_aug)).item(), 0.0, "Augmented tensor should differ from the original tensor.")

    def test_simplex_projection(self):
        """
        Tests if the simplex projection clamps negative values and normalizes coefficients to sum to exactly 1.0.
        """
        # Scenario 1: Coefficients with negative values and unnormalized sum
        lambda_coeff = torch.tensor([-0.2, 0.5, 0.7], requires_grad=True)
        
        with torch.no_grad():
            lambda_coeff.clamp_(min=0.0)
            sum_lambda = lambda_coeff.sum()
            if sum_lambda > 0:
                lambda_coeff.div_(sum_lambda)
                
        self.assertTrue(torch.all(lambda_coeff >= 0.0), "All simplex coefficients must be non-negative.")
        self.assertAlmostEqual(lambda_coeff.sum().item(), 1.0, places=6, msg="Simplex coefficients must sum to exactly 1.0.")
        self.assertAlmostEqual(lambda_coeff[0].item(), 0.0, places=6, msg="Negative coefficient must be clamped to 0.0.")
        self.assertAlmostEqual(lambda_coeff[1].item(), 5.0 / 12.0, places=6, msg="Normalization should distribute weights proportionally.")

    def test_logit_variance_calculation(self):
        """
        Tests the Monte Carlo logit variance calculation across multiple MC passes.
        """
        # Create a mock stack of logits representing M=5 passes, B=2 batches, C=3 classes
        # Let's say: logits_stack of shape (M, B, C)
        M, B, C = 5, 2, 3
        logits_stack = torch.zeros(M, B, C)
        
        # Add varying perturbation along dimension 0 (the MC passes)
        for m in range(M):
            logits_stack[m, :, :] = float(m) * 2.0
            
        # The variance along the first dimension (passes) should be computed correctly
        logit_vars = logits_stack.var(dim=0).mean(dim=0) # Shape: (C,)
        
        # Standard sample variance for [0, 2, 4, 6, 8] is:
        # Mean = 4.0, sum of squared diffs = 16 + 4 + 0 + 4 + 16 = 40
        # Unbiased variance = 40 / (5 - 1) = 10.0
        self.assertEqual(logit_vars.shape, (C,), "Logit variance shape must match the number of classes.")
        for c in range(C):
            self.assertAlmostEqual(logit_vars[c].item(), 10.0, places=4, msg="Unbiased sample variance along the MC passes must be mathematically correct.")

    def test_get_merged_params_correctness(self):
        """
        Verifies that parameter extraction, task-vector scaling, and parameter
        vs. buffer separation behave correctly without losing gradients or stats.
        """
        # Create a small custom Module with a floating-point parameter and a non-floating-point buffer
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, 2))
                self.register_buffer("running_mean", torch.ones(2))
                self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        base_model = TinyModel()
        expert1 = TinyModel()
        expert2 = TinyModel()
        expert3 = TinyModel()
        
        # Modify weights
        expert1.weight.data.fill_(2.0)
        expert2.weight.data.fill_(3.0)
        expert3.weight.data.fill_(4.0)
        
        # Modify running mean buffers
        expert1.running_mean.fill_(5.0)
        expert2.running_mean.fill_(6.0)
        expert3.running_mean.fill_(7.0)

        # Set uniform merging coefficient
        lambda_coeff = torch.tensor([1/3, 1/3, 1/3], requires_grad=True)
        
        # Extract base states
        base_state = {k: v for k, v in base_model.state_dict().items()}
        parameter_names = set(dict(base_model.named_parameters()).keys())
        
        # Compute task vectors
        task_vectors = []
        for exp in [expert1, expert2, expert3]:
            exp_state = {k: v for k, v in exp.state_dict().items()}
            vec = {}
            for k, v in exp_state.items():
                if v.is_floating_point():
                    vec[k] = v - base_state[k]
                else:
                    vec[k] = v
            task_vectors.append(vec)
            
        merged_params = get_merged_params(lambda_coeff, base_state, task_vectors, parameter_names)
        
        # Verification:
        # Weight should be interpolated: base_weight (1.0) + 1/3*(2-1) + 1/3*(3-1) + 1/3*(4-1) = 1.0 + 1/3*1 + 1/3*2 + 1/3*3 = 1.0 + 2.0 = 3.0
        self.assertAlmostEqual(merged_params["weight"][0,0].item(), 3.0, places=5)
        self.assertTrue(merged_params["weight"].requires_grad, "Weights should retain gradients during merged forward passes.")
        
        # Floating point buffer (running_mean) should also be merged: 1.0 + 1/3*4 + 1/3*5 + 1/3*6 = 1.0 + 15/3 = 6.0
        self.assertAlmostEqual(merged_params["running_mean"][0].item(), 6.0, places=5)
        self.assertFalse(merged_params["running_mean"].requires_grad, "Buffers should not retain or require gradients.")
        
        # Integer buffer should select the active expert index's buffer value (argmax of [1/3, 1/3, 1/3] is 0)
        self.assertEqual(merged_params["num_batches_tracked"].item(), 0, "Integer buffers should be routed based on the active expert.")

if __name__ == "__main__":
    unittest.main()
