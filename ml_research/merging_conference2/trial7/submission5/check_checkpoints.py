import torch
checkpoint = torch.load("./checkpoints/expert_mnist.pt", map_location="cpu")
print("Keys in checkpoint:", checkpoint.keys())
print("Keys in state_dict (sample):", list(checkpoint["state_dict"].keys())[:10])
print("Keys in head_state_dict:", list(checkpoint["head_state_dict"].keys()))
