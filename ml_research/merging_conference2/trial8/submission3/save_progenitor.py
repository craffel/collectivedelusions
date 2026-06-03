import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def main():
    print("Loading pretrained ResNet-18 progenitor...")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Modify the classification head to have 10 output classes, matching the fine-tuned experts
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    print("Saving base progenitor checkpoint to progenitor.pt...")
    torch.save(model.state_dict(), "progenitor.pt")
    print("Progenitor saved successfully.")

if __name__ == "__main__":
    main()
