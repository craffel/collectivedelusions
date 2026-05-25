import unittest
import torch
from test_time_merging import project_simplex_batch

class TestSimplexProjection(unittest.TestCase):
    def test_sum_to_one(self):
        # Create arbitrary random vectors
        v = torch.randn(10, 3) * 5.0
        projected = project_simplex_batch(v)
        
        # Check that sum of each projected vector is 1.0
        sums = projected.sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_non_negativity(self):
        # Create arbitrary random vectors with large negative numbers
        v = torch.randn(20, 5) * 10.0 - 5.0
        projected = project_simplex_batch(v)
        
        # Check that all elements are non-negative
        self.assertTrue(torch.all(projected >= 0.0))

    def test_already_on_simplex(self):
        # Vector already on the simplex
        v = torch.tensor([[0.6, 0.3, 0.1]])
        projected = project_simplex_batch(v)
        
        # Check that it is unchanged
        for p, original in zip(projected[0], v[0]):
            self.assertAlmostEqual(p.item(), original.item(), places=5)

    def test_known_projection(self):
        # Known projection case
        v = torch.tensor([[2.0, 0.0, 0.0]])
        projected = project_simplex_batch(v)
        
        # [2.0, 0.0, 0.0] should project to [1.0, 0.0, 0.0]
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        for p, exp in zip(projected[0], expected[0]):
            self.assertAlmostEqual(p.item(), exp.item(), places=5)

if __name__ == '__main__':
    unittest.main()
