import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from experiment import merge_models, apply_fnbc, run_task_specific_bn_calibration

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 32 * 32, 10)

class NetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8 * 32 * 32, 10)
        
    def forward(self, x):
        return self.fc(self.bn(self.conv(x)).view(x.size(0), -1))

class TestModelMerging(unittest.TestCase):
    def test_ties_and_dare_merging(self):
        progenitor = SimpleNet()
        expert1 = SimpleNet()
        expert2 = SimpleNet()
        
        # Modify some parameters slightly
        with torch.no_grad():
            expert1.conv.weight += 0.1
            expert2.conv.weight -= 0.1
            
        # Verify TIES merging
        merged_ties = merge_models(progenitor, [expert1, expert2], method='TIES', lam=0.5)
        self.assertIsNotNone(merged_ties)
        self.assertEqual(merged_ties.conv.weight.shape, progenitor.conv.weight.shape)
        
        # Verify DARE merging
        merged_dare = merge_models(progenitor, [expert1, expert2], method='DARE', lam=0.5)
        self.assertIsNotNone(merged_dare)
        self.assertEqual(merged_dare.conv.weight.shape, progenitor.conv.weight.shape)

    def test_apply_fnbc(self):
        expert1 = NetWithBN()
        expert2 = NetWithBN()
        
        # Modify weights to introduce scale differences
        with torch.no_grad():
            expert1.conv.weight.copy_(expert1.conv.weight * 2.0)
            expert2.conv.weight.copy_(expert2.conv.weight * 0.5)
            
        # Create a merged model (simple Weight Averaging)
        merged = NetWithBN()
        with torch.no_grad():
            merged.conv.weight.copy_((expert1.conv.weight + expert2.conv.weight) / 2.0)
            merged.bn.running_var.copy_((expert1.bn.running_var + expert2.bn.running_var) / 2.0)
            
        original_var = merged.bn.running_var.clone()
        apply_fnbc(merged, [expert1, expert2])
        
        # After FNBC, running_var should be scaled based on the weight norms
        self.assertFalse(torch.allclose(original_var, merged.bn.running_var))
        self.assertTrue((merged.bn.running_var <= original_var).all() or (merged.bn.running_var >= original_var).all())

    def test_run_task_specific_bn_calibration(self):
        model = NetWithBN()
        
        # Generate some dummy data
        dummy_x = torch.randn(10, 3, 32, 32)
        dummy_y = torch.randint(0, 10, (10,))
        dataset = TensorDataset(dummy_x, dummy_y)
        loader = DataLoader(dataset, batch_size=2)
        
        # Randomize running stats
        with torch.no_grad():
            model.bn.running_mean.fill_(99.0)
            model.bn.running_var.fill_(99.0)
            
        run_task_specific_bn_calibration(model, loader, num_batches=3, device='cpu')
        
        # The running stats should be updated and not equal to the filled value 99.0
        self.assertFalse(torch.allclose(model.bn.running_mean, torch.ones_like(model.bn.running_mean) * 99.0))
        self.assertFalse(torch.allclose(model.bn.running_var, torch.ones_like(model.bn.running_var) * 99.0))

if __name__ == '__main__':
    unittest.main()