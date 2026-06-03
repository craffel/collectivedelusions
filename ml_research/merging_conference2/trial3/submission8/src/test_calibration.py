import unittest
import torch
import torch.nn as nn
from models import MultiTaskResNet18
from eval import assemble_merged_model, calibrate_sp_taac, register_sp_taac_hooks

class TestMergingAndCalibration(unittest.TestCase):
    
    def setUp(self):
        # Create a mock model and save checkpoints
        self.device = 'cpu'
        self.mock_pretrained = MultiTaskResNet18(pretrained=False)
        self.mock_expert1 = MultiTaskResNet18(pretrained=False)
        self.mock_expert2 = MultiTaskResNet18(pretrained=False)
        self.mock_expert3 = MultiTaskResNet18(pretrained=False)
        
        # Save mock models to a separate test directory to avoid overwriting trained experts
        import os
        os.makedirs('checkpoints_test', exist_ok=True)
        torch.save(self.mock_pretrained.state_dict(), 'checkpoints_test/pretrained.pt')
        torch.save(self.mock_expert1.state_dict(), 'checkpoints_test/expert_mnist.pt')
        torch.save(self.mock_expert2.state_dict(), 'checkpoints_test/expert_fashion.pt')
        torch.save(self.mock_expert3.state_dict(), 'checkpoints_test/expert_cifar.pt')
        
    def test_weight_averaging(self):
        # Test Weight Averaging assembly
        expert_paths = {
            'mnist': 'checkpoints_test/expert_mnist.pt',
            'fashion': 'checkpoints_test/expert_fashion.pt',
            'cifar': 'checkpoints_test/expert_cifar.pt'
        }
        pretrained_path = 'checkpoints_test/pretrained.pt'
        
        merged = assemble_merged_model(expert_paths, pretrained_path, merge_mode='wa')
        self.assertIsNotNone(merged)
        
        # Verify that weights are averaged
        w_exp1 = torch.load(expert_paths['mnist'])['backbone.conv1.weight']
        w_exp2 = torch.load(expert_paths['fashion'])['backbone.conv1.weight']
        w_exp3 = torch.load(expert_paths['cifar'])['backbone.conv1.weight']
        w_merged = merged.backbone.conv1.weight
        
        expected_avg = (w_exp1 + w_exp2 + w_exp3) / 3.0
        torch.testing.assert_close(w_merged, expected_avg, rtol=1e-5, atol=1e-5)
        
    def test_task_arithmetic(self):
        # Test Task Arithmetic assembly
        expert_paths = {
            'mnist': 'checkpoints_test/expert_mnist.pt',
            'fashion': 'checkpoints_test/expert_fashion.pt',
            'cifar': 'checkpoints_test/expert_cifar.pt'
        }
        pretrained_path = 'checkpoints_test/pretrained.pt'
        
        lambda_val = 0.3
        merged = assemble_merged_model(expert_paths, pretrained_path, merge_mode='ta', lambda_val=lambda_val)
        self.assertIsNotNone(merged)
        
        # Verify that task vectors are combined correctly
        w_base = torch.load(pretrained_path)['backbone.conv1.weight']
        w_exp1 = torch.load(expert_paths['mnist'])['backbone.conv1.weight']
        w_exp2 = torch.load(expert_paths['fashion'])['backbone.conv1.weight']
        w_exp3 = torch.load(expert_paths['cifar'])['backbone.conv1.weight']
        w_merged = merged.backbone.conv1.weight
        
        task_v1 = w_exp1 - w_base
        task_v2 = w_exp2 - w_base
        task_v3 = w_exp3 - w_base
        expected_ta = w_base + lambda_val * (task_v1 + task_v2 + task_v3)
        torch.testing.assert_close(w_merged, expected_ta, rtol=1e-5, atol=1e-5)
        
    def test_sp_taac_hooks(self):
        # Test hook registration and correct scaling behavior
        model = MultiTaskResNet18(pretrained=False)
        gammas = {name: 2.5 for name, module in model.backbone.named_modules() if isinstance(module, nn.BatchNorm2d)}
        
        handles = register_sp_taac_hooks(model, gammas)
        self.assertEqual(len(handles), len(gammas))
        
        # Run a forward pass on dummy inputs and verify that BN inputs are scaled
        x = torch.randn(2, 3, 32, 32)
        # We hook into bn1 input
        bn1_inputs = []
        def bn1_hook(module, input):
            bn1_inputs.append(input[0].clone())
            
        model.backbone.bn1.register_forward_pre_hook(bn1_hook)
        
        # Run forward pass through backbone
        _ = model.backbone(x)
        
        # Remove scaling hooks
        for h in handles:
            h.remove()
            
        # Run unscaled forward pass
        bn1_inputs_unscaled = []
        def bn1_hook_unscaled(module, input):
            bn1_inputs_unscaled.append(input[0].clone())
            
        model.backbone.bn1.register_forward_pre_hook(bn1_hook_unscaled)
        _ = model.backbone(x)
        
        # Verify that the scaled input is exactly 2.5 times the unscaled input
        torch.testing.assert_close(bn1_inputs[0], bn1_inputs_unscaled[0] * 2.5, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
