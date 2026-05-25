import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 128),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        embeddings = self.fc(features)
        return embeddings

class MultiTaskCNN(nn.Module):
    def __init__(self, num_tasks=3, num_classes=10):
        super(MultiTaskCNN, self).__init__()
        self.backbone = CNNBackbone()
        self.heads = nn.ModuleList([
            nn.Linear(128, num_classes) for _ in range(num_tasks)
        ])

    def forward(self, x, task_idx):
        embeddings = self.backbone(x)
        logits = self.heads[task_idx](embeddings)
        return logits, embeddings

def merge_backbone(merged_model, experts, lambdas):
    """
    Interpolates the backbone parameters of experts into merged_model using lambdas.
    Also copies the task heads from the experts into merged_model.
    """
    merged_sd = merged_model.backbone.state_dict()
    for name in merged_sd.keys():
        param_sum = 0.0
        for k, expert in enumerate(experts):
            param_sum += lambdas[k] * expert.backbone.state_dict()[name]
        merged_sd[name] = param_sum
    merged_model.backbone.load_state_dict(merged_sd)
    
    # Copy heads from experts to merged_model
    for k, expert in enumerate(experts):
        merged_model.heads[k].load_state_dict(expert.heads[k].state_dict())
