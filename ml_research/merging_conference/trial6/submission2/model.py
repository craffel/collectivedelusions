import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # x shape: (B, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        out = self.fc2(feat)
        return out, feat

def merge_weights_inplace(target_model, expert_state_dicts, coefficients):
    """
    target_model: nn.Module to be modified in-place
    expert_state_dicts: list of state_dicts [sd1, sd2, sd3]
    coefficients: dict of {param_name: PyTorch tensor of shape (K,) which sums to 1}
    """
    with torch.no_grad():
        for name, param in target_model.named_parameters():
            if name in coefficients:
                coef = coefficients[name] # shape (K,)
                merged_param = torch.zeros_like(param)
                for k, sd in enumerate(expert_state_dicts):
                    merged_param += coef[k] * sd[name]
                param.copy_(merged_param)
            else:
                # If a parameter is not in coefficients, average it across experts
                merged_param = torch.zeros_like(param)
                for sd in expert_state_dicts:
                    merged_param += sd[name]
                merged_param /= len(expert_state_dicts)
                param.copy_(merged_param)
