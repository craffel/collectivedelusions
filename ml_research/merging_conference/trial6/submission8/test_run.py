import torch
import unittest
from run_eval import project_simplex, riemannian_gradient_surgery, riemannian_inner_product

class TestRunEval(unittest.TestCase):
    def test_project_simplex(self):
        v = torch.tensor([1.2, 0.4, 0.1])
        projected = project_simplex(v)
        self.assertAlmostEqual(torch.sum(projected).item(), 1.0, places=5)
        self.assertTrue(torch.all(projected >= 0.0))
        
    def test_riemannian_gradient_surgery(self):
        # Setup dummy gradients
        g1 = {'layer1': torch.tensor([1.0, 0.0, -1.0])}
        g2 = {'layer1': torch.tensor([-1.0, 0.0, 1.0])} # conflicting
        
        G = {'layer1': 2.0} # Metric tensor scalar
        
        # Inner product should be negative (conflicting)
        ip = riemannian_inner_product(g1, g2, G)
        self.assertTrue(ip < 0)
        
        # Surgery
        final_grad = riemannian_gradient_surgery([g1, g2], G)
        self.assertIn('layer1', final_grad)
        
        # The sum should be projected and not conflicting
        self.assertTrue(torch.allclose(final_grad['layer1'], torch.zeros(3)))

if __name__ == '__main__':
    unittest.main()
