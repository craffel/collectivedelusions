import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_base_model():
    # Load ImageNet pretrained ResNet-18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return model

def create_expert_model(num_classes=10):
    model = get_base_model()
    # Replace the classification head (model.fc) with a task-specific head
    model.fc = nn.Linear(512, num_classes)
    return model

def load_checkpoint(model, path, device='cpu'):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
