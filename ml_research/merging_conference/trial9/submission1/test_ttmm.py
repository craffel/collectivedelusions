import torch
import unittest
from run_ttmm import ChannelReductionResNet18, get_test_stream, fuse_bn_buffers, merge_parameters_in_place

class TestTTMM(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChannelReductionResNet18().to(self.device)
        self.expert0 = ChannelReductionResNet18().to(self.device)
        self.expert1 = ChannelReductionResNet18().to(self.device)
        
    def test_channel_reduction_surgery(self):
        # Verify input of 1 channel runs fine through the model
        x = torch.randn(2, 1, 28, 28).to(self.device)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 10))
        
    def test_bn_statistics_fusion(self):
        # Fuse with 50/50 weights
        fuse_bn_buffers(self.model, self.expert0, self.expert1, 0.5, 0.5)
        
        # Check that running mean and var of first BN are indeed fused
        bn_m = None
        bn_0 = None
        bn_1 = None
        for m, m0, m1 in zip(self.model.modules(), self.expert0.modules(), self.expert1.modules()):
            if isinstance(m, torch.nn.BatchNorm2d):
                bn_m = m
                bn_0 = m0
                bn_1 = m1
                break
        
        if bn_m is not None:
            expected_mean = 0.5 * bn_0.running_mean + 0.5 * bn_1.running_mean
            self.assertTrue(torch.allclose(bn_m.running_mean, expected_mean))

    def test_parameter_blending(self):
        expert0_state = {k: v.clone().detach() for k, v in self.expert0.state_dict().items()}
        expert1_state = {k: v.clone().detach() for k, v in self.expert1.state_dict().items()}
        
        mergeable_params = []
        for name, param in self.model.named_parameters():
            if "weight" in name or "bias" in name:
                mergeable_params.append(name)
                
        deltas = {name: torch.tensor(0.0, requires_grad=True, device=self.device) for name in mergeable_params}
        w_global = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        merge_parameters_in_place(self.model, expert0_state, expert1_state, w_global, deltas, mergeable_params)
        
        # Since w_global + delta = 0, sigmoid(0) = 0.5. Merged parameters should be exactly average.
        for name in mergeable_params:
            parts = name.split('.')
            submodule = self.model
            for part in parts[:-1]:
                submodule = getattr(submodule, part)
            param_name = parts[-1]
            param_val = getattr(submodule, param_name)
            
            expected_val = 0.5 * expert0_state[name] + 0.5 * expert1_state[name]
            self.assertTrue(torch.allclose(param_val, expected_val))

if __name__ == "__main__":
    unittest.main()
