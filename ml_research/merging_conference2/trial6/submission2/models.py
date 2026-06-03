import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class RoutingContext:
    weights = None      # shape: [B, K]
    ood_gate = None     # shape: [B, 1, 1, 1]
    is_active = True
    num_tasks = 3
    tau = 0.1
    tau_ood = 0.05
    theta = 0.85        # Similarity threshold for OOD gating

class RoutedConv2d(nn.Module):
    def __init__(self, original_conv, num_tasks=3):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        
        # Base weight from merged model
        self.weight = nn.Parameter(original_conv.weight.data.clone())
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.bias = None
            
        # Task-specific low-rank corrections (initialized to zero)
        self.delta_W = nn.ParameterList([
            nn.Parameter(torch.zeros_like(original_conv.weight.data))
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        # Base convolution
        out_base = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        if not RoutingContext.is_active or RoutingContext.weights is None:
            return out_base
            
        # Compute task-specific corrections
        B = x.size(0)
        out_corr = torch.zeros_like(out_base)
        
        # Compute the correction for each task and scale by routing weights and OOD gate
        for k in range(RoutingContext.num_tasks):
            w_k = RoutingContext.weights[:, k].view(B, 1, 1, 1)
            # Run convolution with delta_W_k
            out_k = F.conv2d(x, self.delta_W[k], None, self.stride, self.padding, self.dilation, self.groups)
            out_corr += w_k * out_k
            
        # Scale the combined correction by the elastic OOD gate
        alpha = RoutingContext.ood_gate.view(B, 1, 1, 1)
        return out_base + alpha * out_corr

class RoutedBatchNorm2d(nn.Module):
    def __init__(self, original_bn, num_tasks=3):
        super().__init__()
        self.num_features = original_bn.num_features
        self.eps = original_bn.eps
        self.momentum = original_bn.momentum
        
        # Base BN (initialized with original merged BN params)
        self.base_bn = nn.BatchNorm2d(self.num_features, eps=self.eps, momentum=self.momentum)
        self.base_bn.weight.data.copy_(original_bn.weight.data)
        self.base_bn.bias.data.copy_(original_bn.bias.data)
        self.base_bn.running_mean.data.copy_(original_bn.running_mean.data)
        self.base_bn.running_var.data.copy_(original_bn.running_var.data)
        
        # Task-specific BNs
        self.task_bns = nn.ModuleList([
            nn.BatchNorm2d(self.num_features, eps=self.eps, momentum=self.momentum)
            for _ in range(num_tasks)
        ])
        for bn in self.task_bns:
            bn.weight.data.copy_(original_bn.weight.data)
            bn.bias.data.copy_(original_bn.bias.data)
            bn.running_mean.data.copy_(original_bn.running_mean.data)
            bn.running_var.data.copy_(original_bn.running_var.data)

    def forward(self, x):
        if not RoutingContext.is_active or RoutingContext.weights is None:
            return self.base_bn(x)
            
        B = x.size(0)
        out = torch.zeros_like(x)
        
        # Soft routing across task-specific BNs
        for k in range(RoutingContext.num_tasks):
            w_k = RoutingContext.weights[:, k].view(B, 1, 1, 1)
            out_k = self.task_bns[k](x)
            out += w_k * out_k
            
        return out

class RoutedResNet18(nn.Module):
    def __init__(self, base_model, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        
        import copy
        copied_model = copy.deepcopy(base_model)
        
        # Copy stem and early layers
        self.conv1 = copied_model.conv1
        self.bn1 = copied_model.bn1
        self.relu = copied_model.relu
        self.maxpool = copied_model.maxpool
        self.layer1 = copied_model.layer1
        self.layer2 = copied_model.layer2
        
        # Deeper layers (to be replaced with routed versions)
        self.layer3 = self._convert_block(copied_model.layer3)
        self.layer4 = self._convert_block(copied_model.layer4)
        
        self.avgpool = copied_model.avgpool
        self.fc = copied_model.fc
        
        # Task prototypes (to be populated during calibration)
        self.prototypes = nn.Parameter(torch.zeros(num_tasks, 128)) # layer2 has 128 channels

    def _convert_block(self, block):
        for i, basic_block in enumerate(block):
            # Convert conv1, conv2
            basic_block.conv1 = RoutedConv2d(basic_block.conv1, self.num_tasks)
            basic_block.conv2 = RoutedConv2d(basic_block.conv2, self.num_tasks)
            # Convert bn1, bn2
            basic_block.bn1 = RoutedBatchNorm2d(basic_block.bn1, self.num_tasks)
            basic_block.bn2 = RoutedBatchNorm2d(basic_block.bn2, self.num_tasks)
            
            # Convert downsample if exists
            if basic_block.downsample is not None:
                # Downsample typically has conv and bn
                new_downsample = []
                for module in basic_block.downsample:
                    if isinstance(module, nn.Conv2d):
                        new_downsample.append(RoutedConv2d(module, self.num_tasks))
                    elif isinstance(module, nn.BatchNorm2d):
                        new_downsample.append(RoutedBatchNorm2d(module, self.num_tasks))
                    else:
                        new_downsample.append(module)
                basic_block.downsample = nn.Sequential(*new_downsample)
        return block

    def forward(self, x):
        # Run stem and early layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        a2_feat = self.layer2(x) # Output of layer2
        
        # Compute Routing Weights & OOD gate at Layer 2
        B = x.size(0)
        if RoutingContext.is_active:
            # Average pool features to get 128-d vector
            feat_pooled = F.adaptive_avg_pool2d(a2_feat, (1, 1)).view(B, -1)
            
            # Compute cosine similarity with prototypes
            # Normalize inputs and prototypes
            feat_norm = F.normalize(feat_pooled, p=2, dim=1)
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)
            
            # similarity shape: [B, K]
            sims = torch.matmul(feat_norm, proto_norm.t())
            
            # Softmax to get routing weights
            RoutingContext.weights = F.softmax(sims / RoutingContext.tau, dim=1)
            
            # OOD Gating based on maximum similarity
            max_sims, _ = torch.max(sims, dim=1)
            # Elastic fallback gate using sigmoid
            RoutingContext.ood_gate = torch.sigmoid((max_sims - RoutingContext.theta) / RoutingContext.tau_ood)
        else:
            RoutingContext.weights = None
            RoutingContext.ood_gate = None
            
        # Forward through deeper blocks
        x = self.layer3(a2_feat)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    return model

def merge_models(expert_models):
    merged = get_resnet18()
    merged_state = merged.state_dict()
    
    expert_states = [m.state_dict() for m in expert_models]
    
    for key in merged_state.keys():
        # Average weights across all experts
        stacked = torch.stack([state[key].float() for state in expert_states], dim=0)
        merged_state[key].copy_(torch.mean(stacked, dim=0))
        
    merged.load_state_dict(merged_state)
    return merged
