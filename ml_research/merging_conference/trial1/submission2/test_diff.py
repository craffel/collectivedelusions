import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torch.func import functional_call

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)

# We want to optimize coefficients 'coeffs' and some custom visual.proj parameter
coeffs = nn.Parameter(torch.tensor([0.3, 0.3, 0.3], device=device, dtype=torch.float32))
proj_param = nn.Parameter(torch.randn(768, 512, device=device))

# Suppose we have dummy task vectors for visual.proj and transformer.resblocks.0.attn.in_proj_weight
tv_proj = {
    'task1': torch.randn(768, 512, device=device),
    'task2': torch.randn(768, 512, device=device),
    'task3': torch.randn(768, 512, device=device),
}

# Construct the parameter dictionary for functional_call on model.visual
visual_params = dict(model.visual.named_parameters())
visual_buffers = dict(model.visual.named_buffers())
all_visual_params = {**visual_params, **visual_buffers}

# Override visual.proj (which has key 'proj' inside model.visual)
merged_proj = proj_param + coeffs[0]*tv_proj['task1'] + coeffs[1]*tv_proj['task2'] + coeffs[2]*tv_proj['task3']
all_visual_params['proj'] = merged_proj

# Run forward pass of model.visual
dummy_images = torch.randn(2, 3, 224, 224, device=device)
features = functional_call(model.visual, all_visual_params, (dummy_images,))
loss = features.sum()

# Backward pass
loss.backward()

print("coeffs.grad:", coeffs.grad)
print("proj_param.grad is None?", proj_param.grad is None)
if proj_param.grad is not None:
    print("proj_param.grad norm:", proj_param.grad.norm().item())
