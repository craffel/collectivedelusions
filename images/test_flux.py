import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
print("Pipeline loaded:", type(pipe))
