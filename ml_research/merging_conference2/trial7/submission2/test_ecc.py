import torch
import unittest
import numpy as np

class TestExactCumulativeCalibration(unittest.TestCase):
    def test_ema_variance_trap_vs_ecc(self):
        """
        Verify that:
        1. Constant momentum EMA keeps a bias to the initial states (EMA Variance Trap).
        2. Dynamic momentum w_t = 1 / (t + 1) mathematically eliminates the initial state
           bias and computes the exact cumulative average of batch statistics in one pass.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        # 10 steps of batches with varying true means and variances
        num_steps = 10
        batch_means = [torch.randn(5) * 0.1 + float(i) for i in range(num_steps)]
        batch_vars = [torch.rand(5) * 0.1 + 0.01 for i in range(num_steps)]

        # Run Standard Constant Momentum EMA (m = 0.1)
        m = 0.1
        running_mean_ema = torch.zeros(5)
        running_var_ema = torch.ones(5) # reset to 1.0

        for t in range(num_steps):
            running_mean_ema = (1 - m) * running_mean_ema + m * batch_means[t]
            running_var_ema = (1 - m) * running_var_ema + m * batch_vars[t]

        # Run our Exact Cumulative Calibration (ECC) (w_t = 1 / (t + 1))
        running_mean_ecc = torch.zeros(5) # start with reset state
        running_var_ecc = torch.ones(5)  # start with reset state

        for t in range(num_steps):
            w_t = 1.0 / (t + 1)
            running_mean_ecc = (1 - w_t) * running_mean_ecc + w_t * batch_means[t]
            running_var_ecc = (1 - w_t) * running_var_ecc + w_t * batch_vars[t]

        # Compute exact mathematical cumulative average of batch statistics
        exact_mean = torch.mean(torch.stack(batch_means), dim=0)
        exact_var = torch.mean(torch.stack(batch_vars), dim=0)

        # Assertions
        # 1. ECC must match the exact cumulative average perfectly (within floating-point tolerance)
        torch.testing.assert_close(running_mean_ecc, exact_mean, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(running_var_ecc, exact_var, rtol=1e-5, atol=1e-5)

        # 2. EMA should differ significantly from the exact average due to initial state bias
        mean_diff = torch.abs(running_mean_ema - exact_mean).mean().item()
        var_diff = torch.abs(running_var_ema - exact_var).mean().item()
        
        print(f"\nExact Mean: {exact_mean.numpy()}")
        print(f"ECC Mean:   {running_mean_ecc.numpy()}")
        print(f"EMA Mean:   {running_mean_ema.numpy()}")
        print(f"Mean Difference (EMA vs Exact): {mean_diff:.4f}")
        
        print(f"\nExact Var:  {exact_var.numpy()}")
        print(f"ECC Var:    {running_var_ecc.numpy()}")
        print(f"EMA Var:    {running_var_ema.numpy()}")
        print(f"Var Difference (EMA vs Exact):  {var_diff:.4f}")

        # Ensure EMA is biased and significantly different from the exact average
        self.assertGreater(var_diff, 0.1, "EMA running variance should retain high bias towards initial state 1.0")

if __name__ == '__main__':
    unittest.main()
