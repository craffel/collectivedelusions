import unittest
import torch
import torch.fft as fft
import numpy as np
import copy

# Formula under test: WRSA scaling map
def wrsa_scaling(x, y, c):
    """
    x: merged spectral component (M_M)
    y: target spectral component (M_T)
    c: regularization parameter
    """
    return (x * y) / (x**2 + (c**2) * y**2 + 1e-15)

def fdsa_scaling(x, y, epsilon=1e-5, clamp_val=5.0):
    scaling = y / (x + epsilon)
    return np.clip(scaling, 1.0/clamp_val, clamp_val)

class TestWRSATheorems(unittest.TestCase):
    
    def test_theorem_3_1_bounded_scaling(self):
        """
        Theorem 3.1: WRSA Bounded Scaling Theorem
        The scaling factor must always be bounded by 1 / (2c) for any x, y >= 0 and c > 0.
        """
        c_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        
        # Test over a large grid of x and y values
        x_vals = np.linspace(0, 100, 500)
        y_vals = np.linspace(0, 100, 500)
        
        for c in c_values:
            theoretical_bound = 1.0 / (2.0 * c)
            
            # Check edge cases
            for x in [0.0, 1e-8, 1e-4, 1.0, 10.0, 100.0]:
                for y in [0.0, 1e-8, 1e-4, 1.0, 10.0, 100.0]:
                    gamma = wrsa_scaling(x, y, c)
                    # Use delta/almostEqual to prevent tiny floating point inaccuracies
                    self.assertLessEqual(gamma, theoretical_bound + 1e-7)
            
            # Check random values
            np.random.seed(42)
            random_x = np.random.uniform(0.0, 1000.0, 1000)
            random_y = np.random.uniform(0.0, 1000.0, 1000)
            for x, y in zip(random_x, random_y):
                gamma = wrsa_scaling(x, y, c)
                self.assertLessEqual(gamma, theoretical_bound + 1e-7)
                
    def test_theorem_3_4_lipschitz_stability(self):
        r"""
        Theorem 3.4: Spectral Estimation Lipschitz Stability Theorem
        The normalized scaling function \tilde{\Gamma}(z) = z / (z^2 + c^2) is globally
        Lipschitz continuous on [0, inf) with a Lipschitz constant of exactly 1 / (c^2).
        """
        c_values = [0.1, 0.2, 0.3, 0.5, 1.0]
        
        # Grid of z values
        z_vals = np.linspace(0.0, 20.0, 1000)
        
        for c in c_values:
            lipschitz_const = 1.0 / (c**2)
            
            # Check for all pairs of z_1 and z_2
            for i in range(len(z_vals) - 1):
                z1 = z_vals[i]
                z2 = z_vals[i+1]
                
                gamma_z1 = z1 / (z1**2 + c**2)
                gamma_z2 = z2 / (z2**2 + c**2)
                
                lhs = abs(gamma_z1 - gamma_z2)
                rhs = lipschitz_const * abs(z1 - z2)
                
                self.assertLessEqual(lhs, rhs + 1e-7)
                
            # Random pairs
            np.random.seed(43)
            random_z1 = np.random.uniform(0.0, 100.0, 500)
            random_z2 = np.random.uniform(0.0, 100.0, 500)
            for z1, z2 in zip(random_z1, random_z2):
                gamma_z1 = z1 / (z1**2 + c**2)
                gamma_z2 = z2 / (z2**2 + c**2)
                lhs = abs(gamma_z1 - gamma_z2)
                rhs = lipschitz_const * abs(z1 - z2)
                self.assertLessEqual(lhs, rhs + 1e-7)

    def test_theorem_3_2_stability_under_noise(self):
        """
        Theorem 3.2: Expected mean squared reconstruction error of WRSA is strictly bounded,
        whereas naive FDSA (without hard clipping) has unbounded variance/error under noise near zero.
        """
        # True target magnitude y and true merged magnitude x
        y = 1.0
        x = 0.0  # Sparse regime
        c = 0.3
        
        np.random.seed(44)
        # Additive zero-mean estimation noise
        noises = np.random.normal(0, 0.05, 10000)
        
        # We will check that WRSA expected MSE is bounded
        wrsa_errors = []
        for eta in noises:
            # Estimated merged magnitude must be non-negative
            x_hat = max(0.0, x + eta)
            gamma_wrsa = wrsa_scaling(x_hat, y, c)
            # Reconstructed activation
            x_test = x_hat # simplified test time behavior
            recon_wrsa = gamma_wrsa * x_test
            err_wrsa = (recon_wrsa - y)**2
            wrsa_errors.append(err_wrsa)
            
        mean_wrsa_err = np.mean(wrsa_errors)
        self.assertTrue(np.isfinite(mean_wrsa_err))
        # Ensure it is reasonably bounded
        self.assertLess(mean_wrsa_err, 2.0)

    def test_theorem_3_5_phase_coherence(self):
        """
        Theorem 3.5: Phase Coherence and Spatial Edge Preservation Theorem
        The spatial L2 error between target Y and reconstructed X is minimized
        when the phase is fully coherent (phi = theta). Any phase perturbation increases error.
        """
        # Create a synthetic 1D signal (or 2D) representing activations
        np.random.seed(45)
        target_spatial = np.random.normal(0, 1, (16, 16))
        target_fft = np.fft.fft2(target_spatial)
        
        target_mag = np.abs(target_fft)
        target_phase = np.angle(target_fft)
        
        # Reconstruct with different phases
        # 1. Coherent phase (phi = theta)
        recon_fft_coherent = target_mag * np.exp(1j * target_phase)
        recon_spatial_coherent = np.real(np.fft.ifft2(recon_fft_coherent))
        
        error_coherent = np.sum((recon_spatial_coherent - target_spatial)**2)
        # Should be near zero because it is a perfect reconstruction
        self.assertAlmostEqual(error_coherent, 0.0, places=10)
        
        # 2. Perturbed phase
        perturbed_phase = target_phase + np.random.uniform(-0.5, 0.5, target_phase.shape)
        recon_fft_perturbed = target_mag * np.exp(1j * perturbed_phase)
        recon_spatial_perturbed = np.real(np.fft.ifft2(recon_fft_perturbed))
        
        error_perturbed = np.sum((recon_spatial_perturbed - target_spatial)**2)
        # Error must be strictly greater than error_coherent
        self.assertGreater(error_perturbed, error_coherent)

    def test_spectral_calibration_hook_integration(self):
        """
        Verify that our SpectralCalibrationHook PyTorch code modifies activations
        correctly in the forward pass.
        """
        from calibrate import SpectralCalibrationHook
        
        # Create dummy activation map [B, C, H, W]
        B, C, H, W = 2, 3, 8, 8
        dummy_act = torch.randn(B, C, H, W)
        
        # Set a flat scaling map of 2.0 (equivalent to scaling spatial output by 2.0)
        scaling_map = torch.ones(H, W) * 2.0
        
        hook = SpectralCalibrationHook(scaling_map)
        calibrated_act = hook.hook_fn(None, None, dummy_act)
        
        # The output must be exactly 2.0 times the input
        expected_act = dummy_act * 2.0
        
        # Verify close parity
        self.assertTrue(torch.allclose(calibrated_act, expected_act, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
