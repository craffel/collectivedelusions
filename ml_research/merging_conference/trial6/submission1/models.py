import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class ResNet18_32x32(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_32x32, self).__init__()
        # Load a base ResNet-18 architecture
        self.resnet = models.resnet18(weights=None)
        # Adapt conv1 for 32x32 resolution (as standard practice in CIFAR benchmarks)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        # Extract features right before the classifier head
        feats = self.resnet.conv1(x)
        feats = self.resnet.bn1(feats)
        feats = self.resnet.relu(feats)
        feats = self.resnet.maxpool(feats)

        feats = self.resnet.layer1(feats)
        feats = self.resnet.layer2(feats)
        feats = self.resnet.layer3(feats)
        feats = self.resnet.layer4(feats)

        feats = self.resnet.avgpool(feats)
        feats = torch.flatten(feats, 1)
        
        if return_features:
            return feats
            
        out = self.resnet.fc(feats)
        return out

    def extract_features(self, x):
        return self.forward(x, return_features=True)

def get_resnet18_32x32():
    return ResNet18_32x32(num_classes=10)

def merge_models_weight_space(experts, coefficients):
    """
    Merges a list of expert models in weight space.
    coefficients: 
        - If a list/tensor of size K: represents global coefficients.
        - If a dict mapping layer names to lists/tensors of size K: represents layer-wise coefficients.
    Returns: A new merged ResNet18_32x32 model with weights merged.
    """
    merged_model = get_resnet18_32x32()
    merged_state_dict = OrderedDict()
    
    # Get the state dict of the first expert as a template
    ref_state_dict = experts[0].state_dict()
    
    for key in ref_state_dict.keys():
        if isinstance(coefficients, dict):
            # Layer-wise merging coefficients
            coeff = coefficients.get(key, None)
            if coeff is None:
                # Fall back to uniform if not found
                coeff = torch.tensor([1.0 / len(experts)] * len(experts), device=ref_state_dict[key].device)
        else:
            # Global merging coefficients
            coeff = coefficients
            
        # Ensure coeff is a tensor on the same device
        if not isinstance(coeff, torch.Tensor):
            coeff = torch.tensor(coeff)
        coeff = coeff.to(ref_state_dict[key].device)
        
        # Merge weight tensors
        merged_val = torch.zeros_like(ref_state_dict[key], dtype=torch.float32)
        for k, expert in enumerate(experts):
            merged_val += coeff[k] * expert.state_dict()[key].to(torch.float32)
            
        merged_state_dict[key] = merged_val.to(ref_state_dict[key].dtype)
        
    merged_model.load_state_dict(merged_state_dict)
    return merged_model
