import torch
import torch.nn as nn
import copy
from main import run_fwmm_shrinkage

print("Mocking inputs...")
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulates a backbone returning 512 features
        self.conv = nn.Conv2d(3, 512, 3, padding=1)
        self.fc = nn.Identity()
    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=(2,3))) # returns [B, 512]

# Create models
calibrated_backbone = MockModel()
expert_backbones = [MockModel(), MockModel()]
heads = {
    "task1": nn.Linear(512, 10),
    "task2": nn.Linear(512, 10)
}
calibration_sets = {
    "task1": torch.randn(4, 3, 32, 32),
    "task2": torch.randn(4, 3, 32, 32)
}

print("Testing run_fwmm_shrinkage on CPU with N=4...")
try:
    cal_heads = run_fwmm_shrinkage(
        calibrated_backbone, 
        expert_backbones, 
        heads, 
        calibration_sets, 
        N=4, 
        N0=16, 
        use_shift=True
    )
    print("Success! Heads calibrated:", list(cal_heads.keys()))
    # Check that shapes and weights are correct and have no NaNs
    for task_name, head in cal_heads.items():
        assert head.weight.shape == (10, 512)
        assert not torch.isnan(head.weight).any()
        assert not torch.isnan(head.bias).any()
    print("All assertions passed successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
