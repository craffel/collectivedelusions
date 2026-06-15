import sys
sys.path.insert(0, "./env_packages")
import torch
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print("Creating vit_tiny model...")
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
print("Model created successfully!")

print("Loading a checkpoint...")
ckpt = torch.load("checkpoints/vit_tiny_pretrained.pt", map_location=device)
print("Checkpoint loaded!")
