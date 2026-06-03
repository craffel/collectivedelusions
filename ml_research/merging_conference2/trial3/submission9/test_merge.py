import unittest
import torch
import numpy as np
from merge import get_dct_matrix, dct_2d, idct_2d, stdfs_merge_tensor

class TestSpectralMerging(unittest.TestCase):
    def setUp(self):
        # Set up a reproducible seed
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dct_reconstruction(self):
        # Verify that 2D-DCT followed by Inverse 2D-DCT reconstructs the original matrix exactly
        shape = (16, 32)
        X = torch.randn(shape, device=self.device)
        
        # Transform
        D, C_M, C_N = dct_2d(X)
        self.assertEqual(D.shape, shape)
        
        # Invert
        X_rec = idct_2d(D, C_M, C_N)
        self.assertEqual(X_rec.shape, shape)
        
        # Verify exact reconstruction within numerical tolerance
        diff = torch.max(torch.abs(X - X_rec))
        self.assertLess(diff.item(), 1e-4)

    def test_stdfs_merge_shape(self):
        # Verify that stdfs_merge_tensor preserves shapes for multiple dimensions
        shapes = [
            (10,),              # 1D tensor
            (8, 16),            # 2D tensor
            (4, 8, 3),          # 3D tensor
            (16, 8, 3, 3)       # 4D tensor (e.g. Conv weight)
        ]
        
        for shape in shapes:
            tensors = [torch.randn(shape, device=self.device) for _ in range(3)]
            merged = stdfs_merge_tensor(tensors, low_freq_ratio=0.1)
            self.assertEqual(merged.shape, shape)
            self.assertEqual(merged.device, self.device)
            self.assertEqual(merged.dtype, tensors[0].dtype)

    def test_stdfs_extremes(self):
        # Verify behavior at extreme low_freq_ratio values (0.0 and 1.0)
        shape = (8, 8)
        tensors = [torch.randn(shape, device=self.device) for _ in range(3)]
        
        # ratio = 1.0 should be exactly equivalent to standard linear averaging (WA)
        merged_1 = stdfs_merge_tensor(tensors, low_freq_ratio=1.0)
        wa = torch.stack(tensors, dim=0).mean(dim=0)
        diff = torch.max(torch.abs(merged_1 - wa))
        self.assertLess(diff.item(), 1e-4)

    def test_dare_merge(self):
        from merge import dare_merge_tensor
        shape = (10, 10)
        tensors = [torch.randn(shape, device=self.device) for _ in range(3)]
        # drop_rate = 0.0 should be equivalent to WA
        merged = dare_merge_tensor(tensors, drop_rate=0.0)
        wa = torch.stack(tensors, dim=0).mean(dim=0)
        diff = torch.max(torch.abs(merged - wa))
        self.assertLess(diff.item(), 1e-4)

        # drop_rate = 0.5 should keep some and zero out some (checked in expected range)
        merged_pruned = dare_merge_tensor(tensors, drop_rate=0.5)
        self.assertEqual(merged_pruned.shape, shape)

    def test_ties_merge(self):
        from merge import ties_merge_tensor
        shape = (10, 10)
        tensors = [torch.randn(shape, device=self.device) for _ in range(3)]
        merged = ties_merge_tensor(tensors, keep_ratio=0.5)
        self.assertEqual(merged.shape, shape)

    def test_s_dare_merge(self):
        from merge import s_dare_merge_tensor
        shape = (8, 16)
        tensors = [torch.randn(shape, device=self.device) for _ in range(3)]
        merged = s_dare_merge_tensor(tensors, drop_rate=0.2)
        self.assertEqual(merged.shape, shape)

    def test_s_ties_merge(self):
        from merge import s_ties_merge_tensor
        shape = (8, 16)
        tensors = [torch.randn(shape, device=self.device) for _ in range(3)]
        merged = s_ties_merge_tensor(tensors, keep_ratio=0.2)
        self.assertEqual(merged.shape, shape)

    def test_stdfs_mask(self):
        # Test that the spectral partition mask selects the correct fraction of coefficients
        from merge import stdfs_merge_tensor
        shape = (16, 16)
        # Verify for ratio = 0.25
        tensors = [torch.randn(shape, device=self.device) for _ in range(3)]
        merged = stdfs_merge_tensor(tensors, low_freq_ratio=0.25)
        self.assertEqual(merged.shape, shape)

    def test_lasp_adaptive_ratio(self):
        from merge import get_adaptive_low_freq_ratio
        # Verify that get_adaptive_low_freq_ratio matches the expected empirical spectral energy ratios and decays monotonically
        ratio_conv = get_adaptive_low_freq_ratio("conv1.weight", base_ratio=0.5)
        ratio_l1 = get_adaptive_low_freq_ratio("layer1.0.conv1.weight", base_ratio=0.5)
        ratio_l2 = get_adaptive_low_freq_ratio("layer2.0.conv1.weight", base_ratio=0.5)
        ratio_l3 = get_adaptive_low_freq_ratio("layer3.0.conv1.weight", base_ratio=0.5)
        ratio_l4 = get_adaptive_low_freq_ratio("layer4.0.conv1.weight", base_ratio=0.5)
        
        # Verify monotonic decay as depth increases: conv1 > layer1 > layer2 > layer3 > layer4
        self.assertGreater(ratio_conv, ratio_l1)
        self.assertGreater(ratio_l1, ratio_l2)
        self.assertGreater(ratio_l2, ratio_l3)
        self.assertGreater(ratio_l3, ratio_l4)
        
        # Verify scaling behavior with base_ratio=0.5
        # raw empirical: conv1=0.2457, layer4=0.1135
        self.assertAlmostEqual(ratio_conv, 0.2457 * 0.5, places=5)
        self.assertAlmostEqual(ratio_l4, 0.1135 * 0.5, places=5)

    def test_swsm_merge(self):
        from merge import swsm_merge_tensor
        shape = (8, 16)
        tensors = [torch.randn(shape, device=self.device) for _ in range(3)]
        
        # SWSM with gamma=0.0 should be equivalent to standard STDFS
        from merge import stdfs_merge_tensor
        merged_swsm_g0 = swsm_merge_tensor(tensors, low_freq_ratio=0.1, gamma=0.0)
        merged_stdfs = stdfs_merge_tensor(tensors, low_freq_ratio=0.1)
        
        diff = torch.max(torch.abs(merged_swsm_g0 - merged_stdfs))
        self.assertLess(diff.item(), 1e-4)
        
        # Check SWSM preserves shape
        merged_swsm_g05 = swsm_merge_tensor(tensors, low_freq_ratio=0.1, gamma=0.05)
        self.assertEqual(merged_swsm_g05.shape, shape)

if __name__ == "__main__":
    unittest.main()
