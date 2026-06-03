import torch
import unittest
import numpy as np
from src.merge import get_task_vectors, ties_merging, dare_merging, calibrate_model

class TestMergeCalibration(unittest.TestCase):
    def setUp(self):
        # Create dummy models (as state dicts)
        self.progenitor = {
            "layer1.weight": torch.ones(5, 10) * 0.1,
            "layer1.bias": torch.zeros(5)
        }
        self.expert1 = {
            "layer1.weight": torch.ones(5, 10) * 0.5,  # task vector is 0.4
            "layer1.bias": torch.ones(5) * 0.2
        }
        self.expert2 = {
            "layer1.weight": torch.ones(5, 10) * 0.3,  # task vector is 0.2 (mean target is 0.3)
            "layer1.bias": torch.ones(5) * (-0.1)
        }
        self.expert_states = [self.expert1, self.expert2]

    def test_task_vectors(self):
        task_vectors = get_task_vectors(self.expert_states, self.progenitor)
        self.assertEqual(len(task_vectors), 2)
        # Check task vectors computation
        # W_k - W_init
        self.assertTrue(torch.allclose(task_vectors[0]["layer1.weight"], torch.ones(5, 10) * 0.4))
        self.assertTrue(torch.allclose(task_vectors[1]["layer1.weight"], torch.ones(5, 10) * 0.2))

    def test_ties_merging(self):
        task_vectors = get_task_vectors(self.expert_states, self.progenitor)
        # Apply TIES-Merging
        merged_tv = ties_merging(task_vectors, self.progenitor, reset_thresh=20)
        # Zeros should not be produced since inputs are positive,
        # but the shapes should be intact.
        self.assertEqual(merged_tv["layer1.weight"].shape, (5, 10))

    def test_qr_sc_wcpr_sparsity_preservation(self):
        task_vectors = get_task_vectors(self.expert_states, self.progenitor)
        
        # Manually inject zeros into merged_tv to check sparsity preservation
        merged_tv = {
            "layer1.weight": torch.zeros(5, 10),
            "layer1.bias": torch.zeros(5)
        }
        merged_tv["layer1.weight"][0, 0] = 1.0
        merged_tv["layer1.weight"][1, 1] = -1.5
        
        # Apply our QR-SC-WCPR calibration
        calibrated_state = calibrate_model(
            merged_tv, task_vectors, self.progenitor, "qr_sc_wcpr",
            gamma=2.0, compensation="inverse"
        )
        
        # Check if W_cal = W_init + T_cal
        # Our calibration should only affect the non-zero updates in merged_tv!
        # Thus, if merged_tv was 0 at an index, the calibrated state at that index should equal progenitor
        for r in range(5):
            for c in range(10):
                if (r == 0 and c == 0) or (r == 1 and c == 1):
                    # Active indices, can be calibrated
                    pass
                else:
                    # Pruned indices, must strictly remain at the progenitor weight
                    self.assertAlmostEqual(calibrated_state["layer1.weight"][r, c].item(), self.progenitor["layer1.weight"][r, c].item())

    def test_qr_sc_wcpr_compensation_scaling(self):
        task_vectors = get_task_vectors(self.expert_states, self.progenitor)
        
        # We set 2 out of 10 parameters in row 0 as active (20% active ratio, p_c = 0.20)
        merged_tv = {
            "layer1.weight": torch.zeros(5, 10),
            "layer1.bias": torch.zeros(5)
        }
        merged_tv["layer1.weight"][0, 0] = 1.0
        merged_tv["layer1.weight"][0, 1] = 1.1
            
        # Calibrate with compensation "inverse" (should scale target by 1/sqrt(0.2) = 2.236)
        calibrated_state_comp = calibrate_model(
            merged_tv, task_vectors, self.progenitor, "qr_sc_wcpr",
            gamma=100.0, compensation="inverse"  # use large gamma to avoid any clamping effects in test
        )
        
        # Calibrate with compensation "none"
        calibrated_state_none = calibrate_model(
            merged_tv, task_vectors, self.progenitor, "qr_sc_wcpr",
            gamma=100.0, compensation="none"  # use large gamma to avoid any clamping effects in test
        )
        
        # The calibrated weights should be scaled up by roughly 1/sqrt(0.2) in inverse vs none
        comp_weights = calibrated_state_comp["layer1.weight"] - self.progenitor["layer1.weight"]
        none_weights = calibrated_state_none["layer1.weight"] - self.progenitor["layer1.weight"]
        
        active_comp = comp_weights[0, :2]
        active_none = none_weights[0, :2]
        
        ratio = (active_comp / (active_none + 1e-8)).mean().item()
        # Should be approximately 1/sqrt(0.2) = 2.236
        self.assertAlmostEqual(ratio, 1.0 / np.sqrt(0.2), places=3)

    def test_quantization_clamping_reconstruction_bound(self):
        # 1. Create a dummy continuous weight vector with severe outliers
        # Size n = 1000. Inliers are drawn from standard normal (scale 0.1)
        # 5 outliers of size 15.0 are injected
        np.random.seed(42)
        n = 1000
        w = np.random.normal(0.0, 0.1, n)
        w[0:5] = [15.0, -14.5, 12.0, -11.0, 13.5]
        
        # 2. Compute dynamic Median/MAD threshold
        median = np.median(w)
        mad = np.median(np.abs(w - median))
        gamma = 2.0
        U = median + gamma * mad
        
        # Clip vector
        w_clip = np.clip(w, -U, U)
        
        # 3. Simulate symmetric uniform 8-bit quantization
        # Bin spacing delta = max(|x|) / (2^(b-1) - 1)
        qmax = 127 # INT8
        
        delta_unclamped = np.max(np.abs(w)) / qmax
        q_unclamped = np.clip(np.round(w / delta_unclamped), -qmax, qmax) * delta_unclamped
        mse_unclamped = np.mean((w - q_unclamped) ** 2)
        
        delta_clipped = np.max(np.abs(w_clip)) / qmax
        q_clipped = np.clip(np.round(w_clip / delta_clipped), -qmax, qmax) * delta_clipped
        mse_clipped = np.mean((w - q_clipped) ** 2)
        
        # 4. Verify that the clipped quantized representation has significantly lower MSE
        # for inlier parameters because the outliers do not inflate the bin size delta
        outliers_mask = np.abs(w) > U
        inliers_mask = ~outliers_mask
        
        mse_inliers_unclamped = np.mean((w[inliers_mask] - q_unclamped[inliers_mask]) ** 2)
        mse_inliers_clipped = np.mean((w[inliers_mask] - q_clipped[inliers_mask]) ** 2)
        
        # Dynamic clamping should reduce inlier quantization noise variance by orders of magnitude!
        self.assertLess(mse_inliers_clipped, mse_inliers_unclamped)
        
        # 5. Verify the theoretical MSE bound from Theorem 4.2:
        # MSE(w, Q(w_clip)) <= delta_clipped^2 / 12 + 1/n * sum_{i: |w_i| > U} (|w_i| - U)^2
        clipping_distortion = np.sum((np.abs(w[outliers_mask]) - U) ** 2) / n
        quantization_noise_bound = (delta_clipped ** 2) / 12.0
        theoretical_bound = quantization_noise_bound + clipping_distortion
        
        # The empirical MSE should be strictly less than or equal to this theoretical bound (with some float margin)
        self.assertLessEqual(mse_clipped, theoretical_bound + 1e-5)

if __name__ == "__main__":
    unittest.main()
