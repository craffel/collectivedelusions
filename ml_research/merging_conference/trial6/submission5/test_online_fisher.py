import unittest
import torch
import torch.nn as nn

class TestOnlineFisherEstimation(unittest.TestCase):
    def test_ema_update_formula(self):
        # Verify the exponential moving average update math
        gamma_fish = 0.1
        f_scalar = 1.0
        
        # Simulated mean squared gradient
        grad_mean_sq = 0.5
        
        # Formula: (1 - gamma_fish) * f_scalar + gamma_fish * grad_mean_sq
        expected_f_scalar = (1 - gamma_fish) * f_scalar + gamma_fish * grad_mean_sq
        
        # Update
        updated_f_scalar = (1 - gamma_fish) * f_scalar + gamma_fish * grad_mean_sq
        
        self.assertAlmostEqual(updated_f_scalar, expected_f_scalar, places=6)
        self.assertAlmostEqual(updated_f_scalar, 0.95, places=6)

    def test_preconditioning_rate_calculation(self):
        # Formula: eta_w = lr * ((f_scalar + eps) ** (-alpha))
        # and clipped at 1.0
        lr = 0.01
        eps = 1e-6
        alpha = 1.0
        
        # Test case 1: small sensitivity (should result in larger learning rate, capped at 1.0)
        f_scalar_small = 1e-8
        eta_w_small = lr * ((f_scalar_small + eps) ** (-alpha))
        eta_w_small_clipped = min(eta_w_small, 1.0)
        
        # (1e-8 + 1e-6)^(-1) = 1e6
        # lr * 1e6 = 10000 -> clipped to 1.0
        self.assertAlmostEqual(eta_w_small_clipped, 1.0, places=5)
        
        # Test case 2: high sensitivity (should result in smaller learning rate)
        f_scalar_large = 100.0
        eta_w_large = lr * ((f_scalar_large + eps) ** (-alpha))
        eta_w_large_clipped = min(eta_w_large, 1.0)
        
        # (100.0)^(-1) = 0.01
        # lr * 0.01 = 0.0001
        self.assertAlmostEqual(eta_w_large_clipped, 0.0001, places=5)

    def test_entropy_gradient_computation(self):
        # Create a simple toy model with one parameter to check active gradients on prediction entropy
        torch.manual_seed(42)
        model_param = nn.Parameter(torch.tensor([0.1, -0.2, 0.5], requires_grad=True))
        
        # Input batch size 2, 3 classes
        logits = model_param.unsqueeze(0).repeat(2, 1) # Shape: (2, 3)
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        
        # Backpropagate entropy with respect to parameter
        grad = torch.autograd.grad(entropy, model_param)[0]
        
        # Ensure gradient is computed and has same shape
        self.assertIsNotNone(grad)
        self.assertEqual(grad.shape, model_param.shape)
        # Prediction entropy is non-zero, and gradient should be non-zero
        self.assertNotEqual(grad.abs().sum().item(), 0.0)

    def test_bias_variance_numerical_simulation(self):
        # Verify Theorem 2: Bias-Variance Trade-Off of Batch-Wise Fisher Estimation
        # Parameters
        mu = 0.5
        sigma = 1.0
        B = 4
        num_trials = 50000
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Simulate gradients
        # Shape: (num_trials, B)
        g_samples = torch.randn(num_trials, B) * sigma + mu
        
        # Batch-average gradient
        # Shape: (num_trials,)
        g_B = g_samples.mean(dim=-1)
        
        # Batch-wise estimator
        g_B_sq = g_B ** 2
        
        # Theoretical values
        true_F_star = mu ** 2 + sigma ** 2 # 0.25 + 1.0 = 1.25
        theoretical_expected = mu ** 2 + (sigma ** 2) / B # 0.25 + 0.25 = 0.50
        theoretical_bias = - ((B - 1) / B) * (sigma ** 2) # -0.75
        theoretical_variance = 2 * (sigma ** 4) / (B ** 2) + 4 * (mu ** 2) * (sigma ** 2) / B # 2/16 + 4*(0.25)/4 = 0.125 + 0.25 = 0.375
        
        # Empirical values
        empirical_mean = g_B_sq.mean().item()
        empirical_bias = empirical_mean - true_F_star
        empirical_variance = g_B_sq.var(unbiased=True).item()
        
        # Verify unbiased-mean expectation
        self.assertAlmostEqual(empirical_mean, theoretical_expected, delta=0.01)
        # Verify bias theorem
        self.assertAlmostEqual(empirical_bias, theoretical_bias, delta=0.01)
        # Verify variance theorem
        self.assertAlmostEqual(empirical_variance, theoretical_variance, delta=0.01)

if __name__ == '__main__':
    unittest.main()
