import torch
from diffusers import DiffusionPipeline
print("Loading pipe")
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
print("Loaded successfully")
