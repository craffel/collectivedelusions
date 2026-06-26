import torch
from diffusers import FluxPipeline
print("Loading pipe")
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda:0")
print("Loaded on cuda:0")
