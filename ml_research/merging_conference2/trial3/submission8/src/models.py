import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Standard ResNet-18 with ImageNet pre-trained weights
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the final fully connected layer with Identity to extract 512-dim features
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Task-specific classification heads outputting 10 logits each
        self.heads = nn.ModuleDict({
            'mnist': nn.Linear(num_features, 10),
            'fashion': nn.Linear(num_features, 10),
            'cifar': nn.Linear(num_features, 10)
        })
        
    def forward(self, x, task):
        # Extract features through the backbone
        features = self.backbone(x)
        # Classify using the task-specific head
        return self.heads[task](features)
        
    def get_backbone_state_dict(self):
        """Return the state dict of the backbone part only."""
        return self.backbone.state_dict()
        
    def load_backbone_state_dict(self, state_dict):
        """Load state dict into the backbone part only."""
        self.backbone.load_state_dict(state_dict)
