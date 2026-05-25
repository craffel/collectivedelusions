import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskResNet18(nn.Module):
    def __init__(self, num_tasks=3, num_classes=10):
        super().__init__()
        # Load pre-trained ResNet-18 as encoder
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # We replace the fc layer with an Identity module to make it a pure feature extractor
        self.resnet.fc = nn.Identity()
        
        # Define separate task-specific classification heads
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_tasks)
        ])
        
    def forward(self, x, task_id):
        # Forward pass on a specific task
        features = self.resnet(x)
        logits = self.heads[task_id](features)
        return logits

def get_base_state_dict(model):
    """
    Returns the encoder state dict of the pre-trained model.
    """
    return {k: v.clone().detach() for k, v in model.resnet.state_dict().items()}

def reconstruct_merged_parameters(base_params, task_vectors, lambdas):
    """
    Reconstructs merged encoder parameters differentiably with respect to lambdas.
    base_params: dict of encoder tensors
    task_vectors: list of dict of encoder tensors (one per task expert)
    lambdas: 1D tensor of shape (num_tasks,) with gradients
    """
    merged_params = {}
    for name, base_param in base_params.items():
        # Correctly merge batchnorm running statistics as direct weighted averages of experts
        if "running_mean" in name or "running_var" in name:
            sum_update = 0
            lambdas_detached = lambdas.detach()
            sum_lambdas = lambdas_detached.sum() + 1e-12
            for k in range(len(task_vectors)):
                weight = lambdas_detached[k] / sum_lambdas
                # Reconstruct expert value from task vector: expert = task_vector + base_param
                expert_val = task_vectors[k][name] + base_param
                sum_update = sum_update + weight * expert_val.to(base_param.device)
            merged_params[name] = sum_update
        elif "num_batches_tracked" in name:
            merged_params[name] = base_param
        else:
            # Standard task vector merging for weights/biases
            sum_update = 0
            for k in range(len(task_vectors)):
                sum_update = sum_update + lambdas[k] * task_vectors[k][name].to(base_param.device)
            merged_params[name] = base_param + sum_update
    return merged_params
