import unittest
import torch
import torch.nn as nn
import os
import sys

# Add current directory to path to import run_experiments_tuned
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from run_experiments_tuned import compute_cka, get_task_indices, load_model

class TestModelMergingAndEvaluation(unittest.TestCase):
    
    def test_compute_cka_identical(self):
        # Generate random representations
        torch.manual_seed(42)
        X = torch.randn(10, 5)
        # CKA with itself should be 1.0
        cka_val = compute_cka(X, X)
        self.assertAlmostEqual(cka_val, 1.0, places=4)
        
    def test_compute_cka_orthogonal(self):
        # Generate orthogonal vectors
        X = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
        Y = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0]])
        # Standardizing means centered. HSIC on orthogonal matrices
        cka_val = compute_cka(X, Y)
        # Centering orthogonal columns may yield small values
        self.assertTrue(cka_val >= 0.0 and cka_val <= 1.0)

    def test_get_task_indices(self):
        class MockDataset:
            def __init__(self):
                self.targets = [0, 5, 10, 15, 20, 25, 9, 11]
                
        dataset = MockDataset()
        # Task 0 has classes 0-9
        indices_task0 = get_task_indices(dataset, task_id=0, num_classes_per_task=10)
        # Expected: target classes 0, 5, 9 (indices 0, 1, 6)
        self.assertEqual(indices_task0, [0, 1, 6])
        
        # Task 1 has classes 10-19
        indices_task1 = get_task_indices(dataset, task_id=1, num_classes_per_task=10)
        # Expected: target classes 10, 15, 11 (indices 2, 3, 7)
        self.assertEqual(indices_task1, [2, 3, 7])

    def test_load_model(self):
        model = load_model()
        self.assertIsInstance(model, nn.Module)
        # Check fc layer shape
        self.assertEqual(model.fc.out_features, 100)

if __name__ == "__main__":
    unittest.main()
