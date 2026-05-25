import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(3136, 128)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ClassifierHead(nn.Module):
    def __init__(self):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        return self.fc(x)

class MergedModel(nn.Module):
    def __init__(self, experts_state_dicts, base_state_dict, num_experts=3):
        """
        experts_state_dicts: list of encoder state dicts for each expert
        base_state_dict: encoder state dict for the pre-trained base model
        """
        super(MergedModel, self).__init__()
        self.encoder = CNNEncoder()
        self.num_experts = num_experts
        self.experts_state_dicts = experts_state_dicts
        self.base_state_dict = base_state_dict
        
        # Identify name of all merging tensors
        # In our case: conv1.weight, conv1.bias, conv2.weight, conv2.bias, conv3.weight, conv3.bias, fc.weight, fc.bias
        self.tensor_names = [name for name, _ in self.encoder.named_parameters()]
        
        # Initialize raw coefficients Lambda of shape [num_tensors, num_experts] with zeros (uniform softmax)
        self.raw_lambdas = nn.Parameter(torch.zeros(len(self.tensor_names), num_experts))
        
    def get_merged_state_dict(self):
        # Softmax over expert coefficients for each tensor
        softmax_lambdas = F.softmax(self.raw_lambdas, dim=1)
        
        merged_sd = {}
        for l_idx, name in enumerate(self.tensor_names):
            merged_tensor = 0.0
            for j in range(self.num_experts):
                weight_j = softmax_lambdas[l_idx, j]
                expert_tensor = self.experts_state_dicts[j][name]
                merged_tensor += weight_j * expert_tensor
            merged_sd[name] = merged_tensor
        return merged_sd

    def forward(self, x, head, active_head_idx=None):
        # Perform differentiable forward pass using the merged state dict
        merged_sd = self.get_merged_state_dict()
        features = torch.func.functional_call(self.encoder, merged_sd, x)
        logits = head(features)
        return logits

